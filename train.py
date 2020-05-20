import json
import argparse
from argparse import Namespace

from tqdm import tqdm
import numpy as np
import scipy.sparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from datasets import data_factory
from models import model_factory
from lib import coarsening, graph, utils
from utils import train_utils, model_utils

manual_seed = 2020
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)


class Trainer(object):

    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        dataset = data_factory[args.data]
        dataset.build_graph(args)
        self.train_dataset = dataset('train')
        self.val_dataset = dataset('val')
        self.test_dataset = dataset('test')
        self.train_loader = DataLoader(self.train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        self.test_loader = DataLoader(self.test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

        classes_num = self.train_dataset.classes_num
        laplacians = dataset.laplacians
        self.model = model_factory[args.filter](laplacians, classes_num, args)
        self.model.apply(model_utils.weight_init)
        self.model.cuda()

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=4, verbose=True)
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = train_utils.EarlyStopping(patience=10)
        self.mean_accuracy = train_utils.MeanAccuracy(classes_num)
        self.mean_loss = train_utils.MeanLoss(args.batch_size)

        print('---'*20)
        print('Model architecture:')
        print('==='*20)
        total_parameters = 0
        for name, param in self.model.named_parameters():
            total_parameters += param.nelement()
            print('{:15}\t{:25}\t{:5}'.format(name, str(param.shape), param.nelement()))
        print('==='*20)
        print('Total parameters: {}'.format(total_parameters))
        print('---'*20)

    def run(self):
        for epoch in range(self.args.max_epochs):
            self.train()
            acc, mloss = self.validation(epoch)
            is_best, is_terminate = self.early_stopping(acc)
            if is_terminate:
                break
            if is_best:
                state_dict = self.model.state_dict()
            self.lr_scheduler.step(mloss)
        
        self.model.load_state_dict(state_dict)
        self.test()

    def train(self):
        self.model.train()
        desc = "TRAINING - loss: {:.4f}"
        pbar = tqdm(total=len(self.train_loader), leave=False, desc=desc.format(0))
        for step, batch in enumerate(self.train_loader):
            data, labels = batch[0].cuda(), batch[1].cuda()
            logits = self.model(data)
            loss = self.criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.desc = desc.format(loss)
            pbar.update(1)
        pbar.close()

    def validation(self, epoch):
        self.model.eval()
        self.mean_loss.reset()
        self.mean_accuracy.reset()
        desc = "VALIDATION - loss: {:.4f}"
        pbar = tqdm(total=len(self.val_loader), leave=False, desc=desc.format(0))
        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                data, labels = batch[0].cuda(), batch[1]
                logits = self.model(data)
                loss = self.criterion(logits, labels.cuda())
                probs = F.softmax(logits, dim=-1).cpu().detach().numpy()
                labels = labels.numpy()
                self.mean_loss.update(loss.cpu().detach().numpy())
                self.mean_accuracy.update(probs, labels)
                pbar.desc = desc.format(loss)
                pbar.update(1)
        pbar.close()
        acc = self.mean_accuracy.compute()
        mloss = self.mean_loss.compute()
        tqdm.write(f"Validation Results - Epoch: {epoch} acc: {acc:.4f} loss: {mloss:.4f}")
        return acc, mloss

    def test(self):
        self.model.eval()
        self.mean_accuracy.reset()
        pbar = tqdm(total=len(self.test_loader), leave=False, desc='TESTING')
        with torch.no_grad():
            for step, batch in enumerate(self.test_loader):
                data, labels = batch[0].cuda(), batch[1]
                logits = self.model(data)
                probs = F.softmax(logits, dim=-1).cpu().detach().numpy()
                labels = labels.numpy()
                self.mean_accuracy.update(probs, labels)

                pbar.update(1)
        pbar.close()
        acc = self.mean_accuracy.compute()
        print(f"Testing Results - acc: {acc:.4f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='20news', choices=['mnist', '20news'])
    parser.add_argument('--filter', type=str, default='fourier', choices=['fourier', 'chebyshev'])
    parser.add_argument('--gc_layers', type=int, default=1, choices=[1, 2])
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = parser.parse_args()
    config = json.load(open('config.json'))[args.data]
    arch_cfg = config['arch'][args.filter]['layer{}'.format(args.gc_layers)]
    graph_cfg = config['graph']
    args = vars(args)
    args.update(arch_cfg)
    args.update(graph_cfg)
    args = Namespace(**args)
    print(args) 

    trainer = Trainer(args)
    trainer.run()