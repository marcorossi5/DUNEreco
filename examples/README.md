# Examples

This directory contains examples scripts.

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
