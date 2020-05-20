import math

import scipy
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from lib import graph
from utils import model_utils

class PolyGCLayer(nn.Module):
    def __init__(self, in_channels, out_channels, poly_degree, pooling_size, laplacian):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        self.pooling_size = pooling_size
        self.poly_degree = poly_degree
        self.laplacian = laplacian

        self.reset_parameters()

    def reset_parameters(self):
        model_utils.truncated_normal_(self.weight, mean=0.0, std=0.1)
        model_utils.truncated_normal_(self.bias, mean=0.0, std=0.1)

    def chebyshev(self, x):
        batch_size, node_num, in_features = x.size()
        out_features = self.weight.size(1)
        x0 = x.permute(1, 2, 0) # node_num x in_features x batch_size
        x0 = torch.reshape(x0, [node_num, in_features * batch_size])
        x_list = [x0]
        if self.poly_degree > 1:
            x1 = torch.sparse.mm(self.laplacian, x0)
            x_list.append(x1)
        for k in range(2, self.poly_degree):
            x2 = 2 * torch.sparse.mm(self.laplacian, x1) - x0  # node_num x in_features*batch_size
            x_list.append(x2)
            x0, x1 = x1, x2
        x = torch.stack(x_list, dim=0) # poly_degree x node_num x in_features*batch_size
        x = torch.reshape(x, [self.poly_degree, node_num, in_features, batch_size])
        x = x.permute(3, 1, 2, 0)  # batch_size x node_num x in_features x poly_degree
        x = torch.reshape(x, [batch_size*node_num, in_features*self.poly_degree])
        x = torch.matmul(x, self.weight)  # batch_size*node_num x out_features
        x = torch.reshape(x, [batch_size, node_num, out_features])  # batch_size x node_num x out_features
        return x

    def brelu(self, x):
        """Bias and ReLU. One bias per filter."""
        return F.relu(x + self.bias)

    def pool(self, x):
        """Max pooling of size p. Should be a power of 2."""
        if self.pooling_size > 1:
            x = x.permute(0, 2, 1)  # batch_size x out_features x node_num
            x = F.max_pool1d(x, kernel_size=self.pooling_size, stride=self.pooling_size)
            x = x.permute(0, 2, 1)  # batch_size x node_num x out_features
        return x

    def forward(self, x):
        x = self.chebyshev(x)
        x = self.brelu(x)
        x = self.pool(x)
        return x


class ChebyshevCGCNN(nn.Module):
    def __init__(self, laplacians, classes_num, args):
        super(ChebyshevCGCNN, self).__init__()
        filter_size = args.filter_size
        pooling_size = args.pooling_size
        poly_degree = args.poly_degree

        laplacians = self.select_laplacian(laplacians, pooling_size)
        laplacians = [self.laplacian_to_sparse(l) for l in laplacians]
        nodes_num = [l.shape[0] for l in laplacians]

        # node_num x out_features x in_features
        flatten_size = nodes_num[args.gc_layers] * filter_size[args.gc_layers-1]
        if args.gc_layers == 1:
            self.gc = PolyGCLayer(1*poly_degree[0], filter_size[0], poly_degree[0], pooling_size[0], laplacians[0])
            self.fc = nn.Linear(in_features=flatten_size, out_features=classes_num)
        elif args.gc_layers == 2:
            self.gc = nn.Sequential(
                PolyGCLayer(1*poly_degree[0], filter_size[0], poly_degree[0], pooling_size[0], laplacians[0]),
                PolyGCLayer(filter_size[0]*poly_degree[1], filter_size[1], poly_degree[1], pooling_size[1], laplacians[1])
            )
            self.fc = nn.Sequential(
                nn.Linear(in_features=flatten_size, out_features=args.hidden_size),
                nn.Dropout(args.dropout),
                nn.Linear(in_features=args.hidden_size, out_features=classes_num)
            )

    def select_laplacian(self, laplacians, pooling_size):
        # Keep the useful Laplacians only. May be zero.
        i = 0
        selected_laplas = [laplacians[0]]
        for p in pooling_size:
            i += int(np.log2(p)) if p > 1 else 0
            selected_laplas.append(laplacians[i])
        return selected_laplas

    def laplacian_to_sparse(self, laplacian):
        # Rescale Laplacian and store as a torch sparse tensor. Copy to not modify the shared laplacian.
        laplacian = scipy.sparse.csr_matrix(laplacian)
        laplacian = graph.rescale_L(laplacian, lmax=2)
        laplacian = laplacian.tocoo()
        indices = torch.LongTensor(np.row_stack((laplacian.row, laplacian.col)))
        data = torch.FloatTensor(laplacian.data)
        shape = torch.Size(laplacian.shape)
        sparse_lapla = torch.sparse.FloatTensor(indices, data, shape).cuda()
        return sparse_lapla

    def forward(self, x):
        # Graph convolutional layers.
        x = torch.unsqueeze(x, 2)
        x = self.gc(x)
        # Fully connected hidden layers.
        batch_size, node_num, feature_num = x.size()
        x = torch.reshape(x, [batch_size, node_num * feature_num])
        x = self.fc(x)
        return x