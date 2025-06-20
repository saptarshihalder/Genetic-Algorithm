# Genetic Algorithm

This repository contains an implementation of an Optimized Performance Based Genetic Algorithm (OPBGA) for scheduling real-time tasks across multiple processors. The algorithm is based on the approach described in the accompanying research paper (see `PAPER 1.pdf`).

## Installation

1. Create a Python environment (optional but recommended).
2. Install dependencies (Numba is used for fast evaluation):
   ```bash
   pip install -r requirements.txt
   ```

## Running the Algorithm

Execute the script to generate a random task set and search for a near optimal schedule:

```bash
python opbga.py
```

The script prints progress information and shows visualisations of the best schedule and convergence curve.

## Tests

Basic tests are provided with `pytest`.
Run them with:

```bash
pytest
```

## Papers

The repository includes two PDF files (`PAPER 1.pdf` and `PAPER 2.pdf`) that describe the underlying scheduling problem and the OPBGA approach.

## Pathology Image Segmentation

`pathology_ga.py` implements an improved Genetic Algorithm for multi-threshold
segmentation of pathology images. The module includes a small example which can
be run directly:

```bash
python pathology_ga.py
```

Running the script will generate a synthetic image, optimise thresholds and
display the resulting segmentation.
