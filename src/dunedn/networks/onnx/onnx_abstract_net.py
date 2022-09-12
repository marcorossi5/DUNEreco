from typing import List
from pathlib import Path
import onnxruntime as ort
from dunedn.training.metrics import MetricsList


class OnnxNetwork(ort.InferenceSession):
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
        super().__init__(ckpt, providers=providers)
        self.metrics = metrics
