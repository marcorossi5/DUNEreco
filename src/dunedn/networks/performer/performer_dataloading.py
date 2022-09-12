import torch
from typing import Tuple
import numpy as np
from dunedn.geometry.helpers import evt2planes, planes2evt


class PlanesDataset(torch.utils.data.Dataset):
    """Loads the dataset for CNN and GCNN networks."""

    def __init__(
        self,
        noisy_planes: np.ndarray,
        clear_planes: np.ndarray = None,
        setup: dict = None,
    ):
        """
        Parameters
        ----------
        noisy_events: np.ndarray
            The noisy events, of shape=(nb wires, nb tdc ticks).
        setup: dict
            The dataset settings dictionary.
        clear_events: np.ndarray
            The event ground truths, if available. Of shape=(nb wires, nb tdc ticks).
        """
        self.noisy_planes = noisy_planes
        self.clear_planes = clear_planes if clear_planes is not None else None
        self.batch_size = setup["batch_size"]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns
        -------
        noisy: torch.Tensor
            A single noisy example, of shape=(1,H,W).
        None
            dummy output for labels.
        """
        if self.clear_planes is not None:
            return self.noisy_planes[index], self.clear_planes[index]
        return self.noisy_planes[index]


def build_dataset_from_events(
    noisy_events: np.ndarray, clear_events: np.ndarray = None, setup: dict = None
) -> Tuple[PlanesDataset, PlanesDataset]:
    noisy_cplanes, noisy_iplanes = evt2planes(noisy_events)
    if clear_events is not None:
        clear_iplanes, clear_cplanes = evt2planes(clear_events)
    else:
        clear_iplanes = None
        clear_cplanes = None
    return PlanesDataset(noisy_iplanes, clear_iplanes, setup), PlanesDataset(
        noisy_cplanes, clear_cplanes, setup
    )
