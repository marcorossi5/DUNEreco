from pathlib import Path
from time import time as tm
import argparse
import torch
from dunedn.networks.models import USCG_Net, GCNN_Net

def get_dummy_input_and_uscg():
    # USCG Net
    h = 100
    w = 500
    pixels = h*w
    dummy_input = torch.arange(pixels).reshape([1, 1, h, w])/ pixels
    return dummy_input, USCG_Net(h=h, w=w)


def get_dummy_input_and_gcnn(modeltype):
    """
    Parameters
    ----------
        - modeltype: str, available options cnn | gcnn
    """
    # GCNN Net
    batch_size = 64
    crop_edge = 32
    # edge_size = (edge_h, edge_w)
    pixels  = batch_size * crop_edge ** 2
    dummy_input =  torch.arange(pixels).reshape([batch_size, 1, crop_edge, crop_edge]) / pixels
    return dummy_input, GCNN_Net(modeltype, "dn", crop_edge, 1, 32)

def get_dummy_input_and_model(modeltype):
    """
    Parameters
    ----------
        - modeltype: str, available options uscg | cnn | gcnn
    
    Returns
    -------
        - torch.Tensor, dummy data for forward pass
        - torch.nn.Module, the model

    Raises
    ------
        - NotImplementedError if modeltype is not in ['uscg', 'cnn', 'gcnn']
    """
    if modeltype == "uscg":
        return get_dummy_input_and_uscg()
    elif modeltype in ["gcnn", "cnn"]:
        return get_dummy_input_and_gcnn(modeltype)
    else:
        raise NotImplementedError("Model not implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modeltype", help="model type, available options: uscg | cnn | gcnn")
    parser.add_argument("--output", type=Path, help="the output onnx model filename", default="network.onnx")
    args = parser.parse_args()
    dummy_input, model = get_dummy_input_and_model(args.modeltype)

    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        dummy_input,
        args.output,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
    )

    # start = tm()
    # model.eval()
    # outputs = model(dummy_input)
    # print(f"Inference done in {tm()-start} s.")
