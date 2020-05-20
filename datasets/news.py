import os

import torch
from torch.utils.data import Dataset 
import numpy as np
from scipy import sparse

from lib import graph, coarsening


class NewsDataset(Dataset):
    perm = None
    graphs = None
    laplacians = None
    
    def __init__(self, split):
        super().__init__()
        if split == 'val':
            split = 'test'
        data_dir = 'data/20news'
        data_path = os.path.join(data_dir, '{}_data.npz'.format(split))
        labels_path = os.path.join(data_dir, '{}_labels.npy'.format(split))
        class_names_path = os.path.join(data_dir, 'class_names.txt')
        self.labels = np.load(labels_path)
        self.class_names = [c.strip() for c in open(class_names_path)]
        self.classes_num = len(self.class_names)

        data = sparse.load_npz(data_path).astype(np.float32)
        self.data = sparse.csr_matrix(coarsening.perm_data(data.toarray(), NewsDataset.perm))

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    @classmethod
    def build_graph(cls, args):
        number_edges = args.number_edges
        metric = args.metric
        normalized_laplacian = args.normalized_laplacian
        coarsening_levels = args.coarsening_levels
        data_dir = 'data/20news'
        embed_path = os.path.join(data_dir, 'embeddings.npy')
        graph_data = np.load(embed_path).astype(np.float32)
        dist, idx = graph.distance_sklearn_metrics(graph_data, k=number_edges, metric=metric)
        adj_matrix = graph.adjacency(dist, idx)
        print("{} > {} edges".format(adj_matrix.nnz//2, number_edges*graph_data.shape[0]//2))
        adj_matrix = graph.replace_random_edges(adj_matrix, 0)
        graphs, perm = coarsening.coarsen(adj_matrix, levels=coarsening_levels, self_connections=False)
        laplacians = [graph.laplacian(g, normalized=normalized_laplacian) for g in graphs]
        cls.perm = perm
        cls.graphs = graphs
        cls.laplacians = laplacians

    @staticmethod
    def collate_fn(batch):
        rows, cols, data = [], [], []
        for i, b in enumerate(batch):
            b_coo = b[0].tocoo()
            rows.extend(b_coo.row + i)
            cols.extend(b_coo.col)
            data.extend(b_coo.data)
        data_batch = sparse.csr_matrix((data, (rows, cols)), shape=(len(batch), b_coo.shape[1]))
        data_batch = torch.from_numpy(data_batch.toarray())
        label_batch = torch.from_numpy(np.array([b[1] for b in batch]))
        return data_batch, label_batch