from pathlib import Path
import torch
from ..gcnn.gcnn_dataloading import GcnnDataset
from .onnx_abstract_net import OnnxNetwork
from .utils import gcnn_onnx_inference_pass
from dunedn.training.metrics import MetricsList


class OnnxGcnnNetwork(OnnxNetwork):
    """Subclass"""

    def __init__(self, ckpt: Path, metrics: MetricsList, providers: list[str] = None):
        """
        Parameters
        ----------
        ckpt: Path
            `.onnx` file path.
        metrics: MetricsList
            List of callable metrics.
        providers: list[str]
            List of providers.
        """
        super().__init__(ckpt, metrics, providers=providers)

    def predict(self, generator: GcnnDataset) -> torch.Tensor:
        """ONNX GCNN network inference.

        Parameters
        ----------
        generator: GcnnDataset
            The inference generator.

        Returns
        -------
        torch.Tensor
            Output tensor of shape=(N,C,H,W).
        """
        return gcnn_onnx_inference_pass(generator, self)
