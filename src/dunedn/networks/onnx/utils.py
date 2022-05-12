import tqdm
import numpy as np
import torch
import onnxruntime as ort
from ..gcnn.gcnn_dataloading import GcnnPlanesDataset


def gcnn_onnx_inference_pass(
    generator: GcnnPlanesDataset, ort_session: ort.InferenceSession
) -> torch.Tensor:
    """
    Parameters
    ----------
    generator: GcnnPlanesDataset
        The inference generator.
    ort_session: ort.InferenceSession
        The onnxruntime inference session.

    Returns
    -------
    torch.Tensor
        Output tensor of shape=(N,C,H,W).
    """
    generator.to_crops()
    test_loader = torch.utils.data.DataLoader(
        dataset=generator,
        batch_size=generator.batch_size,
    )
    outs = []
    for noisy, _ in tqdm.tqdm(test_loader):
        out = ort_session.run(
            None,
            {"input": noisy.numpy().astype(np.float32)},
        )[0]
        outs.append(torch.Tensor(out))
    outs = torch.cat(outs)
    y_pred = generator.converter.tiles2planes(outs)
    generator.to_planes()
    return y_pred
