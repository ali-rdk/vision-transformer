# Vision Transformer (ViT) Implementation

A PyTorch implementation of Vision Transformer for image classification on MNIST dataset.

## Overview

This project implements a Vision Transformer model that processes images by dividing them into patches and applying a transformer-based architecture for classification. The model is trained and evaluated on the MNIST handwritten digit recognition dataset.

## Architecture

- **Patchification**: Images are divided into non-overlapping patches
- **Patch Embedding**: Patches are flattened and mapped to a hidden dimension
- **Positional Embeddings**: Sinusoidal positional encodings added to patch embeddings
- **Transformer Blocks**: Multiple ViT blocks with:
  - Multi-head self-attention mechanism
  - Feed-forward networks (MLP)
  - Layer normalization
  - Dropout regularization
- **Classification Head**: MLP head for final class prediction

## Key Components

- `patchify()`: Converts images into patches
- `get_positional_embeddings()`: Generates positional embeddings
- `Multi_Self_Attention`: Multi-head attention implementation
- `ViT_Block`: Transformer block with attention and MLP
- `ViT_Model`: Complete Vision Transformer model

## Dataset

- **MNIST**: 28×28 grayscale images of handwritten digits (10 classes)
- **Splits**: 80% training, 20% validation, separate test set
- **Batch Size**: 64

## Model Configuration

- Hidden Dimension: 256
- Patch Number: 4×4
- Number of Blocks: 6
- Number of Heads: 8
- Learning Rate: 0.001
- Optimizer: AdamW
- Loss Function: Cross Entropy Loss

## Requirements

```
torch
torchvision
numpy
tqdm
torchinfo
```

## Training

The model is trained for 10 epochs with real-time progress tracking. Training includes:

- Training loss monitoring
- Test accuracy evaluation
- Loss computation per epoch

## Running the Project

Execute the Jupyter notebook `ViT.ipynb` cells sequentially to:

1. Load and prepare the MNIST dataset
2. Define and initialize the ViT model
3. Train the model
4. Evaluate performance on test set

## Results

The model achieves competitive accuracy on MNIST classification task after training.

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
