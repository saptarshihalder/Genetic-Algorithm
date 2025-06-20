# Genetic Algorithm

This repository contains an implementation of an Optimized Performance Based Genetic Algorithm (OPBGA) for scheduling real-time tasks across multiple processors. The algorithm is based on the approach described in the accompanying research paper (see `PAPER 1.pdf`).

## Installation

1. Create a Python environment (optional but recommended).
2. Install dependencies:
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
