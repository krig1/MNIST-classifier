# MNIST Digit Classifier

A handwritten digit classifier built with PyTorch using a fully connected neural network.
Trained and evaluated on the MNIST dataset.

## Architecture

- **Input:** 28x28 grayscale images flattened to 784-dimensional vectors
- **Hidden layers:** 784 → 256 → 128 → 64, each followed by GELU activation
- **Output:** 10 logits (one per digit class)
- **Loss:** CrossEntropyLoss
- **Optimizer:** SGD (lr=0.05)

## How It Works

- Images are normalized using MNIST mean (0.1307) and std (0.3081)
- Forward pass produces raw logits passed into cross-entropy loss
- Backpropagation computes gradients; SGD updates weights each batch
- Model is evaluated on the full 10,000-sample test set after each epoch

## Results

Train Accuracy ≈ 98%
Test Accuracy ≈ 98%

Trained for 5 epochs with batch size 64.

## Dataset

[MNIST](http://yann.lecun.com/exdb/mnist/) — 60,000 training and 10,000 test images of handwritten digits (0-9), loaded via `torchvision.datasets`.

## Dependencies

- torch
- torchvision
