# Denoising models to ONNX

## Documentation and tutorials

ONNX Runtime, an accelerator for machine learning models with multi platform
support. [Docs](https://onnxruntime.ai/docs/)

After getting  an AI a model to be exported, follow one of the
[tutorials](https://onnxruntime.ai/docs/tutorials/) by `ONNX`.
In particular, for PyTorch Inference, refere to these
[docs](https://pytorch.org/docs/stable/onnx.html).

## DUNEdn to ONNX

This folder provides scripts to export and load models in `ONNX` format:

- [save_onnx.py](./save_onnx.py) exports the desired denoising network to `onnx`
format.  
  The following command dumps the desired network to `path.onnx` file:
  
  ```bash
  python save_onnx.py <modeltype> --onnx <path.onnx> --dev <device> [--batch_size BATCH_SIZE]
  ```

  The available choices for `modeltype` are `gcnn` and `cnn`.  
  `uscg` option is
  still not supported by the current version due to a missing op in the `onnx`
  library. Specifically, this is the
  [ADAPTIVEAVGPOOL2D](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html)
  layer used to fix the input resolution of the `scg` layer in the `uscg` network.

  The `--dev` flag needs to be specified to dump the model correctly for a
  specific device. Default il `cpu`, but also `gpu:id` format is supported.

  The optional `--batch_size` flag fixes the input shape of the `ONNX` model.
  Default is `32`. Different array shapes are not allowed during `ONNX Runtime`
  execution, raising an error.

- [load_onnx.py](./load_onnx.py) generates a dummy batch to be passed to both
PyTorch and `onnx` exported networks for inference time comparison.  
  The script runs the inference and prints the performance comparison, measuring
  the speed-up factor provided by `ONNX Runtime` with respect to bare PyTorch
  forward pass.  
  The following command launches the script:

  ```bash
  python load_onnx.py <modeltype> --onnx <path.onnx> [--batch_size BATCH_SIZE]
  ```

  The optional `--batch_size` flag sets the first axis dimension of the
  generated inputs. This should be consistent with the input shape required by
  the model in `ONNX` format.
