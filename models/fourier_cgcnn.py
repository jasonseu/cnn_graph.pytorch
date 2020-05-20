import math

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from lib import graph
from utils import model_utils

class FourierGCLayer(nn.Module):
    def __init__(self, node_num, in_channels, out_channels, pooling_size, U):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(node_num, out_channels, in_channels))
        self.bias = nn.Parameter(torch.Tensor(1, 1, out_channels))
        self.pooling_size = pooling_size
        self.U = U

        self.reset_parameters()

    def reset_parameters(self):
        model_utils.truncated_normal_(self.weight, mean=0.0, std=0.1)
        model_utils.truncated_normal_(self.bias, mean=0.0, std=0.1)

    def filter_in_fourier(self, x):
        batch_size, node_num, in_features = x.size()
        out_features = self.weight.size(1)
        x = x.permute(1, 2, 0)  # node_num x in_features x batch_size
        # Transform to Fourier domain
        x = torch.reshape(x, [node_num, in_features * batch_size])
        x = torch.matmul(self.U, x)
        x = torch.reshape(x, [node_num, in_features, batch_size])
        # Filter
        x = torch.matmul(self.weight, x)  # node_num x out_features x batch_size
        x = x.permute(2, 1, 0)  # batch_size x out_features x node_num
        x = torch.reshape(x, [batch_size * out_features, node_num])
        # Transform back to graph domain
        x = torch.matmul(x, self.U)
        x = torch.reshape(x, [batch_size, out_features, node_num])
        x = x.permute(0, 2, 1)  # batch_size x node_num x out_features
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
        x = self.filter_in_fourier(x)
        x = self.brelu(x)
        x = self.pool(x)
        return x


class FourierCGCNN(nn.Module):
    def __init__(self, laplacians, classes_num, args):
        super(FourierCGCNN, self).__init__()
        filter_size = args.filter_size
        pooling_size = args.pooling_size

        laplacians = self.select_laplacian(laplacians, pooling_size)
        U = [self.get_fourier_basis(l) for l in laplacians]
        nodes_num = [l.shape[0] for l in laplacians]

        # node_num x out_features x in_features
        flatten_size = nodes_num[args.gc_layers] * filter_size[args.gc_layers-1]
        if args.gc_layers == 1:
            self.gc = FourierGCLayer(nodes_num[0], 1, filter_size[0], pooling_size[0], U[0])
            self.fc = nn.Linear(in_features=flatten_size, out_features=classes_num)
        elif args.gc_layers == 2:
            self.gc = nn.Sequential(
                FourierGCLayer(nodes_num[0], 1, filter_size[0], pooling_size[0], U[0]),
                FourierGCLayer(nodes_num[1], filter_size[0], filter_size[1], pooling_size[1], U[1])
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

    def get_fourier_basis(self, laplacian):
        # Fourier basis
        _, U = graph.fourier(laplacian)
        U = torch.tensor(U.T, dtype=torch.float32, requires_grad=False).cuda() # node_num x node_num
        return U

    def forward(self, x):
        # Graph convolutional layers.
        x = torch.unsqueeze(x, 2)
        x = self.gc(x)
        # Fully connected hidden layers.
        batch_size, node_num, feature_num = x.size()
        x = torch.reshape(x, [batch_size, node_num * feature_num])
        x = self.fc(x)
        return x