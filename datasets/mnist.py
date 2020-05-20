import argparse

import torch
import numpy as np
import scipy.sparse
from torchvision.datasets import MNIST
from torch.utils.data import Dataset

from lib import coarsening, graph, utils


class MNISTDataset(Dataset):
    perm = None
    graphs = None
    laplacians = None

    def __init__(self, split):
        super().__init__()
        self.data_dir = './data'
        self.width, self.height = 28, 28
        self.node_num = self.width * self.height

        is_train_or_val = split == 'train' or split == 'val'
        data = MNIST(root=self.data_dir, train=is_train_or_val, download=True)
        self.classes_num = 10
        img_data = data.data.numpy().reshape(-1, self.node_num).astype(np.float32)
        img_labels = data.targets.numpy()

        start = len(img_data) - 5000 if split == 'val' else 0
        end = len(img_data) - 5000 if split == 'train' else len(img_data)
        img_data = img_data[start:end]
        img_labels = img_labels[start:end]
        
        self.img_data = coarsening.perm_data(img_data, MNISTDataset.perm)
        self.img_labels = img_labels

    def __getitem__(self, idx):
        return self.img_data[idx], self.img_labels[idx]

    def __len__(self):
        return self.img_data.shape[0]
    
    @classmethod
    def build_graph(cls, args):
        number_edges = args.number_edges
        metric = args.metric
        normalized_laplacian = args.normalized_laplacian
        coarsening_levels = args.coarsening_levels
        def grid_graph(m, corners=False):
            z = graph.grid(m)
            # compute pairwise distance
            dist, idx = graph.distance_sklearn_metrics(z, k=number_edges, metric=metric)
            A = graph.adjacency(dist, idx) # build adjacent matrix
            # Connections are only vertical or horizontal on the grid.
            # Corner vertices are connected to 2 neightbors only.
            if corners:
                A = A.toarray()
                A[A < A.max()/1.5] = 0
                A = scipy.sparse.csr_matrix(A)
                print('{} edges'.format(A.nnz))

            print("{} > {} edges".format(A.nnz//2, number_edges*m**2//2))
            return A
        
        g = grid_graph(28, corners=False)
        g = graph.replace_random_edges(g, 0)
        graphs, perm = coarsening.coarsen(g, levels=coarsening_levels, self_connections=False)
        laplacians = [graph.laplacian(g, normalized=True) for g in graphs]
        
        cls.perm = perm
        cls.graphs = graphs
        cls.laplacians = laplacians

    @staticmethod
    def collate_fn(batch):
        data_batch = np.array([b[0] for b in batch])
        label_batch = np.array([b[1] for b in batch])
        data_batch = torch.from_numpy(data_batch).type(torch.float32)
        label_batch = torch.from_numpy(label_batch)
        return data_batch, label_batch