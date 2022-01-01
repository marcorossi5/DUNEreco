[![arxiv](https://img.shields.io/badge/arXiv-hep--ph%2F2103.01596-%23B31B1B.svg)](https://arxiv.org/abs/2103.01596)

# DUNEdn

DUNEdn is a denoising algorithm for ProtoDUNE-SP raw data with Neural Networks.

## Installation

The package can be installed with Python's pip package manager:

```bash
git clone https://github.com/marcorossi5/DUNEdn.git
cd DUNEdn
pip install .
export DUNEDN_PATH=$PWD
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

Note: in the current release, the `train` subcommand is not available yet.
It will be issued in the next release.

## Example inference

```bash
dunedn inference -i <input.npy> -o <output.npy> -m <modeltype> [--model_path <checkpoint.pth>]
```

This command takes the `input.npy` array and applies the `modeltype` inference.
A saved model checkpoint could be loaded providing the checkpoint path with the optional `--model_path` flag.

## Saved models

Pretrained models will appear soon in this repository.
