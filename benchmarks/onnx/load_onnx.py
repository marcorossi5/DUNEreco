from pathlib import Path
import argparse
from time import time as tm
import numpy as np
import torch
import onnx
import onnxruntime as ort
from dunedn.networks.models import GCNN_Net, USCG_Net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modeltype", help="model type, available options: uscg | cnn | gcnn")
    parser.add_argument("--onnx", type=Path, help="the saved onnx checkpoint", default="network.onnx")
    parser.add_argument("--check", action="store_true", help="check the onnx model")
    args = parser.parse_args()

    # dummy dataset
    batch_size = 64
    edge_h = 32
    edge_w = 32
    edge_size = (edge_h, edge_w)
    pixels  = edge_h * edge_w * batch_size
    dummy_input = np.arange(pixels).reshape([batch_size, 1, edge_h, edge_w]) / pixels
    print(f"Dummy dataset input shape: {dummy_input.shape}")

    # model import
    if args.modeltype in ["cnn", "gcnn"]:
        model = GCNN_Net(args.modeltype, "dn", edge_h, 1, 32)
    elif args.modeltype == "uscg":
        model = USCG_Net(h=edge_h, w=edge_w)
    else:
        raise NotImplementedError(f"Model {args.modeltype} not implemented")

    if args.check:
        onnx_model = onnx.load(args.onnx.as_posix())
        onnx.checker.check_model(onnx_model)
        print(onnx.helper.printable_graph(onnx_model.graph))

    ort_session = ort.InferenceSession(args.onnx.as_posix())
    start = tm()
    outputs = ort_session.run(
        None,
        {"input": dummy_input.astype(np.float32)},
    )
    onnx_time = tm()-start
    print(f"ONNX Inference with ONNX done in {onnx_time} s.")

    model.eval()
    dummy_input = torch.Tensor(dummy_input)
    start = tm()
    outputs = model(dummy_input)
    pyt_time = tm()-start
    print(f"PyTorch inference done in {pyt_time} s.")
    print(f"ONNX speedup, absolute: {pyt_time/onnx_time:.3}x relative: {100*(pyt_time - onnx_time)/pyt_time:.2}%")