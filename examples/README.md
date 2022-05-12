# Examples

This directory contains examples scripts.

## Note

The examples in this directory assume that the `DUNEdn` package with the
`example` extensions is installed in the current environment.  
This can be done running `pip install -e .[example]` from the package [root](..)
directory.

## Plot example

The plotting example reproduces the analogous images of Fig. 5 from
[arXiv:2103.01596](https://arxiv.org/abs/2103.01596) based on the saved event
sample at
[dunetpc_inspired_p2GeV_cosmics_rawdigits.npy.tar.gz](dunetpc_inspired_p2GeV_cosmics_rawdigits..npy.tar.gz).

In order to run the example on your local machine, `cd` in the current directory
and execute the relative convenience script:

```bash
./run_plot_example.sh
```

This will produce a `plot_example` output folder containing three `.png` images
with the rawdigits image and the two noisy and clear waveforms of a single wire.

## Onnx example

This folder contains a comparison between PyTorch and Onnx model inference.  

The example can be executed as a script with:

```bash
python onnx_example.py
```

And as a [jupyter notebook](onnx_example.ipynb).

## GPU memory usage

GPU memory is precious and OOM errors are extremely annoying.

This section gives an idea of the memory consumption for each model.

### CNN

The CNN Network acts on image crops of `(32,32)` pixels resolution.  
The GPU memory used for a forward pass depends linearly on the batch size:

```text
CNN usage ~ batch_size * 46 MB
```

### GCNN

The GCNN Network acts on image crops of `(32,32)` pixels resolution.  
The GPU memory used for a forward pass depends linearly on the batch size:

```text
GCNN usage ~ batch_size * 84 MB
```

### USCG

The USCG Network typically acts on image windows of different pixels resolution.  
Induction windows are `(800,2000)`, collection ones are `(960,2000)`.

The memory consuptions are listed in the following table

Plane | Resolution | GPU Memory |
|--|--|--|
Induction | 2000x800 | 3.67 GB |
Collection | 2000x960 | 4.40 GB |

The two results are consistent.
