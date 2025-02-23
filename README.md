# Behavior Sequence Transformer for Next Item Prediction

This repository implements a [Behavior Sequence Transformer (BST)](https://arxiv.org/abs/1905.06874) using PyTorch for next item prediction on the MovieLens dataset.

colab：https://colab.research.google.com/drive/1gv3jAHTLgVChAlw7JyRFM5YTGrCX1PIF?usp=sharing


# Overview
The Behavior Sequence Transformer (BST) leverages the Transformer architecture to model user behavior sequences and predict the next item. This project focuses on predicting the next movie a user is likely to watch based on their interaction history.
[bst](/img/bst.png)

# Dataset
The MovieLens dataset is used for training and evaluation. It contains user-movie interactions, including ratings and timestamps. The dataset is preprocessed to create sequential interaction data for next item prediction.

# Installation

1. Clone the repository:
```bash
git clone git@github.com:rintaro121/behavior-sequence-transformer-pytorch.git
```

2. Install dependencies:
```bash
poetry install
```

3. Download data:
```bash
poetry run python dataset/download_ml-10m.py 
```

4. Train the model:
```bash
poetry run python src/main.py
```

5. Visualize results:
```bash
poetry run mlflow ui
```

# Results
[results](/img/results.png)

