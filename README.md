# 🎬 IMDB Sentiment Classifier — Bi-LSTM

A PyTorch-based binary sentiment classifier trained on the IMDB movie review dataset. Uses a bidirectional LSTM with a custom vocabulary to predict whether a review is **positive** or **negative**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Hyperparameters](#hyperparameters)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Inference](#inference)

---

## Overview

This project trains a sentiment analysis model on the [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (50,000 labeled movie reviews). It preprocesses raw HTML-tagged reviews, builds a vocabulary from scratch, and trains a bidirectional LSTM classifier.

**Key features:**
- Custom tokenizer and vocabulary builder (no external NLP libraries needed)
- Bidirectional LSTM with dropout regularization
- 80/10/10 train/val/test split with reproducible seeding
- Automatic best-model checkpointing based on validation accuracy
- Simple `predict()` function for inference on new text

---

## Project Structure

```
.
├── sentiment_train.py       # Main training script (this file)
├── IMDB Dataset.csv         # Dataset (download separately)
├── sentiment_model.pt       # Saved checkpoint (generated after training)
└── README.md
```

---

## Requirements

```
torch
pandas
```

Install with:

```bash
pip install torch pandas
```

> Python 3.8+ recommended. GPU training is supported automatically via CUDA if available.

---

## Dataset

Download the IMDB dataset CSV from Kaggle and place it in the same directory as the script:

- **File:** `IMDB Dataset.csv`
- **Columns:** `review` (text), `sentiment` (`positive` / `negative`)
- **Size:** 50,000 reviews (balanced: 25k positive, 25k negative)

The preprocessing pipeline:
1. Strips HTML tags from review text
2. Lowercases and tokenizes using regex (`\b\w+\b`)
3. Encodes labels: `positive → 1`, `negative → 0`

---

## Model Architecture

```
Input (token IDs, max length 300)
        ↓
Embedding Layer  [vocab_size × 64]
        ↓
Bi-LSTM  [2 layers, hidden=128, dropout=0.3]
        ↓
Concatenate final forward + backward hidden states  [256]
        ↓
Dropout (0.3)
        ↓
Fully Connected Layer  [256 → 1]
        ↓
Sigmoid → Probability score [0, 1]
```

- Reviews longer than 300 tokens are truncated; shorter ones are zero-padded.
- Unknown tokens at inference time are mapped to `<UNK>` (index 1).

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Vocabulary size | 30,000 (+ `<PAD>`, `<UNK>`) |
| Max sequence length | 300 |
| Embedding dim | 64 |
| LSTM hidden dim | 128 |
| LSTM layers | 2 |
| Bidirectional | Yes |
| Dropout | 0.3 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 10 |
| Gradient clipping | 1.0 |

---

## Usage

### Training

Run the script directly:

```bash
python sentiment_train.py
```

The script will:
1. Load and preprocess `IMDB Dataset.csv`
2. Build the vocabulary
3. Split data into train (80%), validation (10%), and test (10%) sets
4. Train for 10 epochs, printing loss and accuracy each epoch
5. Save the best checkpoint to `sentiment_model.pt` whenever validation accuracy improves
6. Print final test accuracy and run a few sample predictions

### Expected Output

```
Using device: cuda
Vocab size:  30002
Train: 40000 | Val: 5000 | Test: 5000

Epoch 01/10 | Train Loss: 0.4821  Train Acc: 0.7689 | Val Loss: 0.3504  Val Acc: 0.8512
  ✓ Checkpoint saved (val_acc=0.8512)
...
Epoch 10/10 | Train Loss: 0.1823  Train Acc: 0.9301 | Val Loss: 0.2741  Val Acc: 0.8934

Test Loss: 0.2810 | Test Acc: 0.8901
Best Val Acc: 0.8934
```

---

## Results

Typical performance after 10 epochs on the IMDB dataset:

| Split | Accuracy |
|---|---|
| Train | ~93% |
| Validation | ~89% |
| Test | ~89% |

---

## Inference

Use the `predict()` function to classify new reviews:

```python
predict("This movie was absolutely fantastic — a masterpiece!", model, vocab)
# Score : 0.9821  ->  Positive 😊

predict("Terrible pacing, awful dialogue, complete waste of time.", model, vocab)
# Score : 0.0312  ->  Negative 😞
```

To load a saved checkpoint for inference:

```python
checkpoint = torch.load("sentiment_model.pt", map_location=device)
vocab      = checkpoint['vocab']
hp         = checkpoint['hyperparams']

model = LSTMClassifier(
    vocab_size     = len(vocab),
    embedding_dim  = hp['embedding_dim'],
    hidden_dim     = hp['hidden_dim'],
    output_dim     = 1,
    n_layers       = hp['n_layers'],
    bidirectional  = hp['bidirectional'],
    dropout        = hp['dropout'],
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

> **Note:** Out-of-vocabulary (OOV) tokens encountered during inference are automatically mapped to `<UNK>` and listed in the output for transparency.

---

## Reproducibility

A fixed random seed (`SEED = 42`) is set for both PyTorch and the dataset split generator, ensuring consistent results across runs.
