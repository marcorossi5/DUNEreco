# Examples

This directory contains examples scripts.

## Note

The examples in this directory assume that the `DUNEdn` package with the
`example` extensions is installed in the current environment.  
This can be done running `pip install -e .[example]` from the package [root](..)
directory.

## Onnx examples

This folder contains a comparison between PyTorch and Onnx model inference.  

The examples can be executed as scripts with:

```bash
python onnx_accuracy_example.py
```

And as a [jupyter notebook](onnx_accuracy_example.ipynb).

The performance examples can be executed in the same manner through the
`onnx_performance_example` files in this folder.

## GPU memory usage

GPU memory is precious and OOM errors are extremely annoying.

This section gives an idea of the memory consumption for each model, so that the
user can decide the best strategy based on the available hardware.

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
Default values for induction windows are `(800,2000)`, while collection ones are
`(960,2000)`.

The memory consuptions are listed in the following table

Plane | Resolution | GPU Memory |
|--|--|--|
Induction | 2000x800 | 3.67 GB |
Collection | 2000x960 | 4.40 GB |

The two results are consistent.
