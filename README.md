# CNN_Graph.pytorch

This repository is an unofficial implememnts of the paper "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" in NIPS 2016 with **PyTorch**. It supports to train and evaluate the network on the 20news and mnist.

## Requirements

- Python 3.6
- PyTorch 1.1

## Train and Evaluation

With following scripts, you can train and evaluate the graph-based CNN with corresponding network architectures on mnist dataset.

```bash
python train.py --data mnist --filter fourier --gc_layers 1
python train.py --data mnist --filter fourier --gc_layers 2
python train.py --data mnist --filter chebyshev --gc_layers 1
python train.py --data mnist --filter chebyshev --gc_layers 2
```
where `gc_layers=1` corresponds to the network architecture `GC32`, and `gc_layers=2` to `GC32-P4-GC64-P4-FC512`
To train and evaluate on 20news dataset, you need to run following script in order to preprocess the documents and generate required intermediate data.

```
python scripts/20news_preprocess.py
```

Then you can train the network with different negtwork architectures on 20news dataset.

```bash
python train.py --data 20news --filter fourier --gc_layers 1
python train.py --data 20news --filter chebyshev --gc_layers 1
```

Note that the codes under the folder `lib` are completely borrowed from original codebase [CNN_Graph](https://github.com/mdeff/cnn_graph), where the redundant functions have been removed. This part is responsible for graph building and coarsening. 

# Performance 

**MNIST**

| filters   | gc_layer=1 | gc_layer=2 |
| --------- | ---------- | ---------- |
| Fourier   |   0.9747   |   0.9788   |
| Chebyshev |   0.9816   |   0.9818   |

**20NEWS**

| filters   | gc_layer=1 | gc_layer=2 |
| --------- | ---------- | ---------- |
| Fourier   |   0.5504   |     -      |
| Chebyshev |   0.5554   |     -      |

# Acknowledgements

Thanks the official [CNN_Graph](https://github.com/mdeff/cnn_graph) implemented with TensorFlow and awesome PyTorch team.