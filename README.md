[![arxiv](https://img.shields.io/badge/arXiv-hep--ph%2F2103.01596-%23B31B1B.svg)](https://arxiv.org/abs/2103.01596)

# DUNEdn

DUNEdn is a denoising algorithm for ProtoDUNE-SP raw data with Neural Networks.

## Installation

The package can be installed with Python's pip package manager:

```bash
git clone https://github.com/marcorossi5/DUNEreco.git
cd dunedn
pip install .
```

This process will copy the dunedn program to your environment python path.

DUNEdn requires the following packages:

- python3
- numpy
- pytorch
- matplotlib
- hyperopt (optional)

## Running the code

In order to launch the code

```bash
dunedn <subcommand> [options]
```

Valid subcommands are: `preprocess|train|inference`.
