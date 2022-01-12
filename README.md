# DUNEdn

[![arxiv](https://img.shields.io/badge/arXiv-hep--ph%2F2103.01596-%23B31B1B.svg)](https://arxiv.org/abs/2103.01596)
[![DOI](https://zenodo.org/badge/248536693.svg)](https://zenodo.org/badge/latestdoi/248536693)

![pytest](https://github.com/N3PDF/pdfflow/workflows/pytest/badge.svg)

If you use this software please cite this [paper](https://doi.org/10.1007/s41781-021-00077-9)

```bibtex
@article{dunedn,
author={Rossi, Marco
and Vallecorsa, Sofia},
title={Deep Learning Strategies for ProtoDUNE Raw Data Denoising},
journal={Computing and Software for Big Science},
year={2022},
month={Jan},
day={07},
volume={6},
number={1},
numpages={9},
issn={2510-2044},
doi={10.1007/s41781-021-00077-9},
url={https://doi.org/10.1007/s41781-021-00077-9}
}
```

DUNEdn is a denoising algorithm for ProtoDUNE-SP raw data with Neural Networks.

## Installation

The package can be installed with Python's pip package manager:

```bash
git clone https://github.com/marcorossi5/DUNEdn.git
cd DUNEdn
pip install .
```

This process will copy the DUNEdn program to your environment python path.

### Note

The [saved_models](saved_models) directory contains the checkpoints to reproduce
the results presented in [arXiv:2103.01596](https://arxiv.org/abs/2103.01596).  
Since some of the saved models files are quite large (~100 MB), they are uploaded
via [git-lfs](https://git-lfs.github.com/). when cloning the repo, it is possible
to download pointers to those large files rather than the whole binaries. This can
be achieved adding the flag `--config lfs.fetchexclude="*.pth*"` to the `git clone`
command above.

### Requirements

DUNEdn requires the following packages:

- python3
- numpy
- pytorch
- torchvision
- matplotlib
- hyperopt

## Running the code

In order to launch the code

```bash
dunedn <subcommand> [options]
```

Valid subcommands are: `preprocess|train|inference`.  
Use `dunedn <subcommand> --help` to print the correspondent help message.  
For example, the help message for `train` subcommand is:

```bash
$ dunedn train --help
usage: dunedn train [-h] [--output OUTPUT] [--force] configcard

Train model loading settings from configcard.

positional arguments:
  configcard       yaml configcard path

optional arguments:
  -h, --help       show this help message and exit
  --output OUTPUT  output folder
  --force          overwrite existing files if present
```

### Configuration cards

Models' parameter settings are stored in configcards. The [configcards](configcards)
folder contains some examples. These can be extended providing the path to user
defined cards directly to the command line interface.

Setting the `DUNEDN_SEARCH_PATH` environment variable it is possible to let DUNEdn
looking for configcards into different directories automatically. More on the
search behavior can be found at the `get_configcard_path` function's docstring
in the [utils/ultis.py](src/dunedn/utils/utils.py) file.

### Preprocess a dataset

At first, a dataset directory should have the following structure:

```text
dataset directory tree structure:
dataset_dir
  |-- train
  |    |--- evts
  |-- val
  |    |--- evts
  |-- test
  |    |--- evts
```

where each `evts` folder contains a collection of ProtoDUNE events stored as raw
digits (numpy array format).

It is possible to generate the correspondent dataset to train an USCG or a GCNN
network with the command:

```bash
dunedn preprocess <configcard.yaml> --dir_name <dataset directory>
```

This will modify the dataset directory tree in the following way:

```txt
dataset directory tree structure:
dataset_dir
  |-- train
  |    |--- evts
  |    |-- planes (preprocess product)
  |    |-- crops (preprocess product)
  |-- val
  |    |--- evts
  |    |--- planes (preprocess product)
  |-- test
  |    |--- evts
  |    |--- planes (preprocess product)
```

### Training a model

After specifying parameters inside a configuration card, leverage DUNEdn to train
the correspondent model with:

```bash
dunedn train <configcard.yaml>
```

The output directory is set by default to `output`. Optionally, the
`DUNEDN_OUTPUT_PATH` environment variable could be set to override this choice.

### Inference

```bash
dunedn inference -i <input.npy> -o <output.npy> -m <modeltype> [--model_path <checkpoint.pth>]
```

DUNEdn inference takes the `input.npy` array and forwards it to the desired model
`modeltype`. The output array is saved to `output.npy`.

If a checkpoint directory path is given with the optional `--model_path` flag, a
saved model checkpoint could be loaded for inference.  
The checkpoint directory should have the following structure:

```text
model_path
    |-- collection
    |       |-- <ckpt directory name>_dn_collection.pth
    |-- induction
    |       |-- <ckpt directory name>_dn_induction.pth
```

On the other hand, if `--model_path` is not specified, an un-trained network is issued.

### Benchmark

The paper results can be reproduced through the
[compute_denoising_performance.py](benchmarks/compute_denoising_performance.py) benchmark.  
Please, see the script's docstring for further information.
