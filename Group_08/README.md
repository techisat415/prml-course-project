# Visual Question Answering on the SHAPES Dataset

A PyTorch-based pipeline for answering compositional boolean queries about synthetic images containing simple geometric shapes.

## Overview

The SHAPES dataset consists of 30×30 RGB images arranged on a 3×3 grid. Each cell may contain one of three **shapes** (circle, square, triangle) in one of three **colors** (red, green, blue). Every image is paired with a compositional query written in a parenthesised notation (e.g. `(is blue (left_of red))`) and a boolean answer (`true` / `false`).

The goal is to predict the correct answer for unseen image–query pairs.

## Approach

Rather than treating VQA as an end-to-end black box, the pipeline decomposes the problem into interpretable steps:

1. **Heuristic cell labelling** — each 10×10 cell of the image is labelled with a shape and colour using pixel-level heuristics (dominant hue, edge/corner patterns).
2. **CNN classifier** — a lightweight convolutional network (`CellClassifier`) is trained on the heuristic labels to predict the shape and colour of each cell more robustly.
3. **Scene graph construction** — the CNN predictions are assembled into a structured scene representation (a 3×3 grid of `{shape, color}` objects).
4. **Query evaluation** — compositional queries are parsed and evaluated against the scene graph using a recursive interpreter, producing a boolean prediction.

## Model Architecture — `CellClassifier`

```
Input: 10×10 RGB cell image
  └─ Conv2d(3 → 32, k=3)  + BatchNorm + ReLU
  └─ Conv2d(32 → 64, k=3) + BatchNorm + ReLU
  └─ Conv2d(64 → 128, k=3)+ BatchNorm + ReLU
  └─ AdaptiveAvgPool2d(1)
       ├─ shape_head: Linear(128 → 3)   → shape logits
       └─ color_head: Linear(128 → 3)   → colour logits
```

The model is trained for **30 epochs** with cross-entropy loss on both heads simultaneously.

## Dataset

| Split  | Images | Queries | Labels |
|--------|-------:|--------:|-------:|
| tiny   |     64 |      64 |     64 |
| small  |    640 |     640 |    640 |
| med    |  6,400 |   6,400 |  6,400 |
| large  | 13,568 |  13,568 | 13,568 |
| test   |  1,024 |   1,024 |  1,024 |

Expected files in the working directory:

```
train.tiny.input.npy   train.tiny.query   train.tiny.output
train.small.input.npy  train.small.query  train.small.output
train.med.input.npy    train.med.query    train.med.output
train.large.input.npy  train.large.query  train.large.output
test.input.npy         test.query         test.output
```

## Requirements

```
python >= 3.10
torch
numpy
matplotlib
tqdm
scikit-learn
```

Install dependencies:

```bash
pip install torch numpy matplotlib tqdm scikit-learn
```

A CUDA-capable GPU is recommended; the notebook auto-detects and falls back to CPU.

## Notebook Structure

| Section | Description |
|---------|-------------|
| **1 — Setup** | Imports, random seeds, device configuration, dataset loading |
| **2 — EDA** | Visualisation of sample images and query distribution |
| **3 — Heuristics** | Pixel-level cell labelling to generate pseudo-labels for CNN training |
| **4 — CNN Training** | `CellClassifier` definition, `CellDataset`, training loop across all data sizes |
| **5 — Evaluation** | Scene graph construction, query parsing & evaluation, metrics (precision, recall, F1, accuracy), comparison plots |
| **6 — Save Outputs** | Persist per-size predictions and best model weights |

## Results

The model is trained and evaluated independently on each training size. Reported on the held-out test set (1,024 examples):

| Training size | Test accuracy |
|---------------|:-------------:|
| tiny          | —             |
| small         | **93.26%**    |
| med           | —             |
| large         | —             |

> The `small` split achieved the best test accuracy (93.26 %, 955/1024) and its weights are saved as `cell_classifier_small.pth`.

Metrics computed: **Precision, Recall, F1, Accuracy** with confusion-matrix breakdown (TP / FP / FN / TN). Results are visualised as grouped bar charts and an F1-vs-training-size line plot.

## Output Files

After running all cells the following files are written to the working directory:

| File | Description |
|------|-------------|
| `predictions_tiny.txt` | Boolean predictions for the test set (tiny model) |
| `predictions_small.txt` | Boolean predictions for the test set (small model) |
| `predictions_med.txt` | Boolean predictions for the test set (med model) |
| `predictions_large.txt` | Boolean predictions for the test set (large model) |
| `cell_classifier_<best_size>.pth` | Saved state dict of the best-performing model |

Each predictions file contains one `true` or `false` per line, matching the order of `test.query`.

## Reproducibility

Fixed seeds are set at the top of the notebook:

```python
np.random.seed(43)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Re-running the notebook end-to-end on the same data will reproduce the reported results.