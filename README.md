# OCoDeDANet

Official implementation of **OCoDeDANet**, a method for detecting **disjoint** and **overlapping communities** in **dynamic node-attributed networks**.

OCoDeDANet combines:
- graph topology,
- node attributes,
- temporal smoothness,
- and automatic relevance determination (ARD)

to identify community structure over time and estimate the effective number of communities.

## Manuscript

This repository accompanies the manuscript:

**Detecting disjoint and overlapping communities in temporal node-attributed networks**  
*Renny Márquez, Richard Weber, Alex Barrales-Araneda*

> Citation details will be added once the paper is publicly available.

## Repository

Source code:  
https://github.com/rennymarquez/OCoDeDANet/

## Installation

Clone the repository:

```bash
git clone https://github.com/rennymarquez/OCoDeDANet.git
cd OCoDeDANet
```

Create and activate a virtual environment.

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Build

The project is packaged into an executable archive using the provided `Makefile`.

Build the executable with:

```bash
make
```

This generates the executable file:

```bash
ocodedanet
```

To remove generated files:

```bash
make clean
```

## Usage

After building the project, run:

```bash
./ocodedanet --help
```

General usage:

```bash
./ocodedanet [-h] --model {DBNMFARD,DUBNMFARD} [--initial-K INITIAL_K]
             [--n-iters N_ITERS] [--min-iters MIN_ITERS] [--a A] [--b B]
             [--tolerance TOLERANCE] [--alpha ALPHA]
             [--matrix-seed MATRIX_SEED] [--csv-file CSV_FILE]
             data
```

## Arguments

### Required
- `data`: path to the input data file
- `--model`: model to apply (`DBNMFARD` or `DUBNMFARD`)

### Optional
- `--initial-K`: initial number of communities
- `--n-iters`: maximum number of iterations
- `--min-iters`: minimum number of iterations
- `--a`: hyperparameter `a`
- `--b`: hyperparameter `b`
- `--tolerance`: convergence tolerance
- `--alpha`: temporal trade-off parameter
- `--matrix-seed`: random seed for initialization
- `--csv-file`: path to save results in CSV format

## Examples

Basic example:

```bash
./ocodedanet --model DBNMFARD data/Data17R0M0S1
```

Example with explicit parameters:

```bash
./ocodedanet --model DBNMFARD --matrix-seed 1234 --initial-K 50 --alpha 1 data/Data123S3
```

## Testing

The provided `Makefile` includes a `test` target. After creating the virtual environment and installing dependencies, tests can be run with:

```bash
make test
```

## Data

The datasets used in the manuscript are referenced in the associated paper, where the public data link is provided.

If you want to run the code with your own datasets, they must follow the input structure expected by the implementation.

## Citation

If you use this repository in academic work, please cite the associated paper once it becomes available.

A BibTeX entry will be added after publication.

## Authors

- Renny Márquez
- Richard Weber
- Alex Barrales-Araneda
