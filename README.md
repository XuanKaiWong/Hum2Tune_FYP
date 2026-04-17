# Hum2Tune

Hum2Tune is an FYP project for humming-based song identification.

## What is fixed in this updated codebase

- file-level train/validation/test splitting to avoid leakage from the same recording appearing in multiple splits
- chunked dataset writing and loading for large feature sets
- consistent dataset format across training and evaluation
- train/evaluate/predict CLI now supports explicit model selection
- cleaned, aligned unit tests
- lighter repo layout without generated artifacts, temporary files, or corrupted dataset chunks

## Current scope

This repository implements **closed-set song identification** from humming clips. It does not perform open-world internet-scale retrieval.

## Recommended workflow

```bash
python main.py setup
python main.py dataset --create
python main.py train --model cnn_lstm
python main.py evaluate --model cnn_lstm
python main.py predict --audio path/to/hum.wav --model cnn_lstm
```

## Data layout

Place humming recordings under:

```text
data/Humming Audio/<song_name>/<recording>.wav
```

Each folder name becomes one class label.

## Important research note

The dataset preparation now splits **by file before window extraction and augmentation**. This is critical for a fair FYP evaluation.

## What is not included in this zip

- raw audio datasets
- processed features
- trained model checkpoints
- generated result plots/logs

Those files were intentionally excluded so the repository is clean and submission-friendly.
