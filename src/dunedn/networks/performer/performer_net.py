import logging
from pathlib import Path
from time import time as tm
import torch
from torch import nn
from typing import List, Tuple
from ..abstract_net import AbstractNet
from .performer_dataloading import PlanesDataset
from .performer_net_blocks import (
    PreProcessBlock,
    HPF,
    LPF,
    PostProcessBlock,
)
from .utils import performer_inference_pass
from dunedn import PACKAGE
from dunedn.geometry.pdune import geometry as pdune_geometry
from dunedn.networks.utils import BatchProfiler

logger = logging.getLogger(PACKAGE + ".performer")


class PerformerNet(AbstractNet):
    def __init__(self, input_channels: int, hidden_channels: int):
        """
        Example
        -------

        >>> import torch
        >>> from dunedn.networks.performer.performer_net import PerformerNet
        >>> model = PerformerNet(1, 32)
        >>> x = torch.randn(1, 1, 64, 64)
        >>> print(model(x).shape)

        """
        super().__init__()
        ic = input_channels
        hc = hidden_channels
        self.pre_process_blocks = nn.ModuleList(
            [
                PreProcessBlock(3, ic, hc),
                PreProcessBlock(5, ic, hc),
                PreProcessBlock(7, ic, hc),
                PreProcessBlock(9, ic, hc),
            ]
        )
        self.lpfs = nn.ModuleList([LPF(7, hc * 4, hc * 4) for _ in range(4)])
        self.hpf = HPF(3, hc * 4, hc * 4)
        self.post_process_block = PostProcessBlock(3, ic, hc)

        self.combine = lambda x, y: x + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PerformerNet forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape=(N,C,H,W).

        Returns
        -------
        output: torch.Tensor
            Output tensor of shape=(N,C,H,W).
        """
        y = torch.cat([block(x) for block in self.pre_process_blocks], dim=1)
        y_hpf = self.hpf(y)
        y = self.combine(y, y_hpf)
        for lpf in self.lpfs:
            y = self.combine(lpf(y), y_hpf)
        output = self.post_process_block(y) * x
        return output

    def predict(
        self,
        generator: PlanesDataset,
        dev: str = "cpu",
        no_metrics: bool = False,
        verbose: int = 1,
        profiler: BatchProfiler = None,
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]], float]:
        """Gcnn network inference.

        Parameters
        ----------
        generator: GcnnDataset
            The inference dataset generator.
        device: str
            The device hosting the computation. Defaults is "cpu".
        no_metrics: bool
            Wether to skip metric computation. Defaults to False, so metrics are
            indeed computed.
        verbose: int
            Switch to log information. Defaults to 1. Available options:

            - 0: no logs.
            - 1: display progress bar.

        profiler: BatchProfiler
            The profiler object to record batch inference time.

        Returns
        -------
        y_pred: torch.Tensor
            Denoised planes, of shape=(N,1,H,W).
        logs: dict
            The computed metrics results in dictionary form.
        """
        self.check_network_is_compiled()

        test_loader = torch.utils.data.DataLoader(
            dataset=generator,
            batch_size=generator.batch_size,
        )

        # inference pass
        start = tm()
        y_pred = performer_inference_pass(
            test_loader, self, dev, verbose, profiler=profiler
        )
        inference_time = tm() - start

        if no_metrics:
            return y_pred

        # compute metrics
        y_true = generator.clear
        logs = self.metrics_list.compute_metrics(y_pred, y_true)
        logs.update({"time": inference_time})
        return y_pred, logs

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        dev: str = "cpu",
    ) -> dict:
        """Trains the network for one epoch.

        Parameters
        ----------
        train_loader: torch.utils.data.DataLoader
            The training dataloader.
        validation_data: torch.utils.data.DataLoader
            The validation dataloader.
        dev: str
            The device hosting the computation.

        Returns
        -------
        epoch_logs: dict
            The dictionary of epoch logs.
        """
        logger.debug("Training epoch")
        self.train()

        for clear, noisy in train_loader:
            self.callback_list.on_train_batch_begin()
            step_logs = self.train_batch(noisy, clear, dev)
            self.callback_list.on_train_batch_end(step_logs)

        epoch_logs = {}
        return epoch_logs

    def train_batch(self, noisy: torch.Tensor, clear: torch.Tensor, dev: str):
        """Makes one batch update.

        Parameters
        ----------
        noisy: torch.Tensor
            Noisy inputs batch, of shape=(B,1,H,W).
        clear: torch.Tensor
            Clear target batch, of shape=(B,1,H,W)
        dev: str
            The device hosting the computation.

        Returns
        -------
        step_logs: dict
            The step logs as a dictionary.
        """
        noisy = noisy.to(dev)
        self.optimizer.zero_grad()
        y_pred = self.forward(noisy)

        loss = self.loss_fn(y_pred, clear)
        loss.mean().backward()
        self.optimizer.step()

        step_logs = self.metrics_list.compute_metrics(y_pred, clear)
        step_logs.update({"loss": loss.item()})
        return step_logs

    def onnx_export(self, fname: Path):
        """Export model to ONNX format.

        Parameters
        ----------
        fname: Path
            The path to save the `.onnx` network.
        """
        # produce dummy inputs
        inputs = torch.randn(
            1, 1, pdune_geometry["nb_tdc_ticks"], pdune_geometry["nb_tdc_ticks"]
        )

        # export network
        torch.onnx.export(
            self,
            inputs,
            fname,
            verbose=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "channels"},
                "output": {0: "batch_size", 2: "channels"},
            },
        )
