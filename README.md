# Code and Data for NBTI Aging Prediction with APPD-Net

This repository provides the code, processed data, and pretrained weights for the experiments reported in our manuscript on NBTI aging prediction using APPD-Net.

The current release includes the inference script, processed datasets, pretrained weights, and instructions for reproducing the reported figures.

## Included content

- `run.py`: inference-only script that loads pretrained weights and generates the figures.
- `vgs=4.csv`: release-version processed data file.
- `T=150.csv`: release-version processed data file.
- `weights_exp156/`: pretrained weights required by the inference script.

## What this repository does

The script loads pretrained weights and processed data, then reproduces the figures for the selected experiments.

## What this repository does not include

- training code
- optimizer setup
- backpropagation or parameter update steps
- full research pipeline outside experiments 1, 5, and 6

## Environment

Recommended Python version: 3.10+

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Repository structure

```text
.
├── run.py
├── vgs=4.csv
├── T=150.csv
├── requirements.txt
├── .gitignore
└── weights_exp156/
```

## How to run

Place the two CSV files and the `weights_exp156` folder in the same directory as the script.
Unzip weights_exp156.zip into the weights_exp156 folder before running the script.

Run:

```bash
python run.py --data-dir . --weights-dir weights_exp156 --figure-dir figures_exp156_pretrained
```

## Output

The script:

1. loads pretrained weights,
2. reads the processed CSV files,
3. generates figures for experiments 1, 5, and 6,
4. saves the figures to `figures_exp156_pretrained`
