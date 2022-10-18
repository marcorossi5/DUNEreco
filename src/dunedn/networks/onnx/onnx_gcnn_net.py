from pathlib import Path
from time import time as tm
from typing import List
import torch
from ..gcnn.gcnn_dataloading import GcnnDataset, TilingDataset
from .onnx_abstract_net import OnnxNetwork
from .utils import gcnn_onnx_inference_pass
from dunedn.networks.utils import BatchProfiler
from dunedn.training.metrics import MetricsList


class OnnxGcnnNetwork(OnnxNetwork):
    """Subclass"""

    def __init__(self, ckpt: Path, metrics: MetricsList, providers: List[str] = None):
        """
        Parameters
        ----------
        ckpt: Path
            `.onnx` file path.
        metrics: MetricsList
            List of callable metrics.
        providers: List[str]
            List of providers.
        """
        super().__init__(ckpt, metrics, providers=providers)

    def predict(
        self,
        generator: TilingDataset,
        no_metrics: bool = False,
        verbose: int = 1,
        profiler: BatchProfiler = None,
    ) -> torch.Tensor:
        """ONNX GCNN network inference.

        Parameters
        ----------
        generator: GcnnDataset
            The inference generator.
        profiler: BatchProfiler
            The profiler object to record batch inference time.

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

        # inference pass
        start = tm()
        output = gcnn_onnx_inference_pass(test_loader, self, verbose, profiler=profiler)
        inference_time = tm() - start

        # convert back to events
        y_pred = generator.converter.crops2image(output).numpy()

        if no_metrics:
            return y_pred

        # compute metrics
        generator.to_events()
        y_true = generator.clear
        logs = self.metrics_list.compute_metrics(y_pred, y_true)
        logs.update({"time": inference_time})
        return y_pred, logs
