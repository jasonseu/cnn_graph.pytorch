from datasets.mnist import MNISTDataset
from datasets.news import NewsDataset

data_factory = {
    'mnist': MNISTDataset,
    '20news': NewsDataset
}