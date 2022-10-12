from pathlib import Path
import torch
from typing import Tuple
import numpy as np
from dunedn.geometry.helpers import evt2planes
from dunedn.utils.utils import median_subtraction


class PlanesDataset(torch.utils.data.Dataset):
    """Loads the dataset for CNN and GCNN networks."""

    def __init__(
        self,
        data_folder: Path,
        batch_size: int,
        should_load_target: bool = False,
    ):
        """
        Parameters
        ----------
        data_folder: Path
            The folder containing the dataset.
        batch_size: int
            The length of the mini batch size dimension.
        should_load_target: np.ndarray
            The event ground truths, if available. Of shape=(nb wires, nb tdc ticks).
        """
        self.batch_size = batch_size
        self.data_folder = data_folder
        noisy_path = data_folder / "collection_noisy.npy"
        noisy_planes = np.load(noisy_path)
        iplanes = np.load(noisy_path.as_posix().replace("collection", "induction"))

        # for training purporses, induction planes are better to be padded to
        # match the collection planes
        # for induction planes first do median subtraction, then pad
        channel_pad = noisy_planes.shape[2] - iplanes.shape[2]
        pad = ((0, 0), (0, 0), (0, channel_pad), (0, 0))
        # the median subtraction op changes the dtype to float64
        pad_sub_iplanes = np.pad(median_subtraction(iplanes), pad)
        # median_subtraction return an array of float64 due to np.median op
        # explicitly cast to float32 (default for pytorch ops)

        self.noisy_planes = np.concatenate([noisy_planes, pad_sub_iplanes], axis=0)
        self.noisy_planes = self.noisy_planes.astype(np.float32)
        self.noisy_planes = self.noisy_planes[..., :128, :128]

        if should_load_target:
            clear_path = data_folder / "collection_clear.npy"
            clear_planes = np.load(clear_path)
            iplanes = np.load(clear_path.as_posix().replace("collection", "induction"))
            pad_sub_iplanes = np.pad(median_subtraction(iplanes), pad)
            pad_sub_iplanes = pad_sub_iplanes.astype(np.float32)
            self.clear_planes = np.concatenate([clear_planes, pad_sub_iplanes], axis=0)
            self.clear_planes = self.clear_planes.astype(np.float32)

            self.clear_planes = self.clear_planes[..., :128, :128]
        else:
            self.clear_planes = None

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

    def __len__(self):
        return len(self.noisy_planes)


# TODO: fix this for inference
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
