"""This module implements dataset loading for the CNN and GCNN networks."""
from abc import ABC, abstractmethod
import logging
from pathlib import Path
import re
from typing import List, Tuple
import numpy as np
import torch
from ..utils import get_hits_from_clear_images
from .gcnn_net_utils import Converter
from dunedn import PACKAGE
from dunedn.geometry.helpers import evt2planes, planes2evt
from dunedn.geometry.pdune import nb_cchannels, nb_ichannels
from dunedn.utils.utils import median_subtraction

logger = logging.getLogger(PACKAGE + ".gcnn")


def float_me(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    elif isinstance(x, torch.Tensor):
        return x.float()
    else:
        return float(x)


def subtract_events(events: List[np.ndarray]) -> List[np.ndarray]:
    """Loads events from file or directory and perform APA median subtraction.

    Parameters
    ----------
    events: List[np.ndarray]
        The list of loaded events array, each of shape=(H,W).

    Returns
    -------
    events: List[np.ndarray]
        The events of subtracted events, each of shape=(H, W).
    """
    iplanes, cplanes = evt2planes(events)
    return planes2evt(median_subtraction(iplanes), median_subtraction(cplanes))


def load_data_from_file(data_path: Path, should_subtract: bool) -> np.ndarray:
    """Loads events from file and perform APA median subtraction.

    Parameters
    ----------
    data_path: Path
        The path pointing to the event file.
    should_subtract: bool
        Wheter to do median subtraction on inputs or not.

    Returns
    -------
    np.ndarray:
        The events array, of shape=(nb_events, 1, H, W).
    """
    events = np.load(data_path)[None, None, :, 2:]
    if should_subtract:
        events = subtract_events(events)
    return events


def load_data_from_folder(
    data_folder: Path, filter_key: str, should_subtract: bool
) -> np.ndarray:
    """Loads events directory and perform APA median subtraction.

    Parameters
    ----------
    data_path: Path
        The path pointing to the dataset folder.
    filter_key: str
        The key discriminating between noisy or clear inputs.
    should_subtract: bool
        Wheter to do median subtraction on inputs or not.

    Returns
    -------
    np.ndarray:
        The events array, of shape=(nb_events, 1, H, W).
    """
    paths = [
        f
        for f in data_folder.iterdir()
        if re.match(filter_key, f.name) and f.suffix == ".npy"
    ]
    events = np.stack([np.load(f) for f in paths], axis=0)[:, None, :, 2:]
    if should_subtract:
        events = subtract_events(events)
    return events


class BaseGcnnDataset(torch.utils.data.Dataset, ABC):
    """Loads the dataset for CNN and GCNN networks."""

    def __init__(
        self,
        dataset_type: str,
        task: str = "dn",
        channel: str = "collection",
        dsetup: dict = None,
        batch_size: int = 128,
    ):
        """
        Parameters
        ----------
        dataset_type: str
            Available options train | val | test
        task: str
            Available options dn | roi.
        channel: str
            Available options induction | collection
        dsetup: dict
            The dataset settings dictionary.
        batch_size: int
            The number of examples to be batched.
        """
        self.dataset_type = dataset_type
        self.task = task
        self.channel = channel
        self.dsetup = dsetup
        self.batch_size = int(batch_size)

        self.training = self.dataset_type == "train"
        self.crop_size = self.dsetup["crop_size"]
        self.threshold = dsetup["threshold"]

    def __len__(self):
        return len(self.noisy)

    @abstractmethod
    def __getitem__(self, index):
        pass


class GcnnDataset(BaseGcnnDataset):
    """Loads the dataset for CNN and GCNN networks."""

    def __init__(
        self,
        dataset_type: str,
        task: str,
        channel: str,
        dsetup: dict,
        batch_size: int,
    ):
        """
        Parameters
        ----------
        dataset_type: str
            Available options train | val | test
        task: str
            Available options dn | roi.
        channel: str
            Available options induction | collection
        dsetup: dict
            The dataset settings dictionary.
        batch_size: int
            The number of examples to be batched.
        """
        super().__init__(dataset_type, task, channel, dsetup, batch_size)

        self.data_folder = self.dsetup["data_folder"] / dataset_type
        self.crops_folder = self.data_folder / "crops"
        self.planes_folder = self.data_folder / "planes"

        crop_edge = self.dsetup["crop_edge"]
        pct = self.dsetup["pct"]

        # if dataset_type is training, load crops. Load planes otherwise.
        if self.training:
            fname = self.crops_folder / f"{channel}_clear_{crop_edge}_{pct}.npy"
            clear = np.load(fname)

            fname = self.crops_folder / f"{channel}_noisy_{crop_edge}_{pct}.npy"
            # median subtraction is made on a plane basis: crops are already
            # normalized during preprocessing stage
            noisy = np.load(fname)
        else:
            fname = self.planes_folder / f"{channel}_clear.npy"
            clear = np.load(fname)

            fname = self.planes_folder / f"{channel}_noisy.npy"
            noisy = np.load(fname)
            noisy = median_subtraction(noisy)
            self.converter = Converter(self.crop_size)

        if self.task == "roi":
            clear = get_hits_from_clear_images(clear, self.threshold)
            self.balance_ratio = np.count_nonzero(clear) / clear.size()

        self.noisy = torch.Tensor(noisy)
        self.clear = torch.Tensor(clear)

    def to_crops(self):
        """Converts planes into crops.

        Note
        ----

        This method should not be called when training.
        """
        if self.training:
            logger.error("`to_crops()` method should not be called when training")

        self.noisy = self.converter.planes2tiles(self.noisy)
        self.clear = self.converter.planes2tiles(self.clear)

    def to_planes(self):
        """Converts crops into planes."""
        if self.training:
            logger.error("`to_planes()` method should not be called when training")
        self.noisy = self.converter.tiles2planes(self.noisy)
        self.clear = self.converter.tiles2planes(self.clear)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        noisy: torch.Tensor
            A single noisy example, of shape=(1,H,W).
        clear: torch.Tensor
            A single clear example, of shape=(1,H,W).
        """
        return self.noisy[index], self.clear[index]


class GcnnPlanesDataset(BaseGcnnDataset):
    """Loads the dataset for CNN and GCNN networks."""

    def __init__(
        self,
        noisy: np.ndarray,
        task: str,
        channel: str,
        dsetup: dict,
        batch_size: int,
    ):
        """
        Parameters
        ----------
        noisy: np.ndarray
            The noisy planes for inference.
        task: str
            Available options dn | roi.
        channel: str
            Available options induction | collection
        dsetup: dict
            The dataset settings dictionary.
        batch_size: int
            The number of examples to be batched.
        """
        super().__init__("test", task, channel, dsetup, batch_size)

        self.converter = Converter(self.crop_size)

        noisy = median_subtraction(noisy)
        self.noisy = torch.Tensor(noisy)

    def to_crops(self):
        """Converts planes into crops.

        Note
        ----

        This method should not be called when training.
        """
        if self.training:
            logger.error("`to_crops()` method should not be called when training")

        self.noisy = self.converter.planes2tiles(self.noisy)

    def to_planes(self):
        """Converts crops into planes."""
        if self.training:
            logger.error("`to_planes()` method should not be called when training")
        self.noisy = self.converter.tiles2planes(self.noisy)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns
        -------
        noisy: torch.Tensor
            A single noisy example, of shape=(1,H,W).
        None
            dummy output for labels.
        """
        return self.noisy[index], 0


class TilingDataset(torch.utils.data.Dataset):
    """Loads data from path and implements tiling procedure.

    If ``data_path`` is a regular file, then this is supposed to be an event.
    Else if  ``data_path`` is a directory, it is supposed to contain files matching
    the ``rawdigit`` string in them.

    The ``has_target`` replaces the matched ``rawdigit`` subtring in the
    matched file names in data path with ``rawdigit_noiseoff``, to identify
    events containing ground truth information.
    """

    def __init__(
        self,
        data_path: Path,
        batch_size: int = 32,
        crop_size: Tuple[int, int] = (32, 32),
        has_target: bool = False,
    ):
        """
        Parameters
        ----------
        data_folder: Path
            The path pointing to the dataset. Either a regular file or a folder.
        batch_size: int
            The length of the mini batch size dimension.
        crop_size: Tuple[int, int]
            The tile dimensions: (edge_h, edge_w).
        has_target: np.ndarray
            The event ground truths, if available. Of shape=(nb wires, nb tdc ticks).
        """
        # super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.has_target = has_target

        # check on crop size: the number of wires on the induction and collection
        # planes must be exactly divisible by the ``crop_size[0]``.
        # This is to avoid wire mixing between plane crops.
        msg = (
            f"induction ({nb_ichannels}) and collection ({nb_cchannels}) planes "
            "number of channels must be exactly divisible by the crop height, "
            f"got {crop_size[0]}"
        )
        assert (
            nb_cchannels % crop_size[0] == 0 and nb_ichannels % crop_size[0] == 0
        ), msg

        self.converter = Converter(self.crop_size)

        if self.data_path.is_file():
            noisy = load_data_from_file(self.data_path, should_subtract=True)
            # store statistics
            self.nb_events = noisy.shape[0]
            if self.has_target:
                clear_path = self.data_path.name.replace(
                    "rawdigit", "rawdigit_noiseoff"
                )
                clear = load_data_from_file(
                    self.data_path.parent / clear_path, should_subtract=False
                )
            else:
                clear = None
        elif self.data_path.is_dir():
            noisy = load_data_from_folder(
                self.data_path,
                filter_key=r"rawdigit(?!_noiseoff)",
                should_subtract=True,
            )
            if self.has_target:
                clear = load_data_from_folder(
                    self.data_path,
                    filter_key=r"rawdigit_noiseoff",
                    should_subtract=False,
                )
            else:
                clear = None
        else:
            raise ValueError(f"not a regular file: {self.data_path}")

        self.yields_events = True
        self.yields_crops = False

        self.noisy = float_me(noisy)
        self.clear = float_me(clear) if clear is not None else None

        self.nb_events = self.noisy.shape[0]
        self.to_crops()
        self.nb_crops = self.noisy.shape[0]

        logger.info(f"Loaded {self.nb_crops} crops from {self.nb_events} events")

    def to_crops(self):
        """Converts planes into crops."""
        assert self.yields_events ^ self.yields_crops
        # if already returning crops, do nothing
        if self.yields_crops:
            return
        self.yields_events = False
        self.yields_crops = True
        self.noisy = self.converter.image2crops(self.noisy)
        if self.has_target:
            self.clear = self.converter.image2crops(self.clear)

    def to_events(self):
        """Converts crops into events."""
        assert self.yields_events ^ self.yields_crops
        # if already returning events, do nothing
        if self.yields_events:
            return
        self.yields_events = True
        self.yields_crops = False
        self.noisy = self.converter.crops2image(self.noisy)
        if self.has_target:
            self.clear = self.converter.crops2image(self.clear)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns data at ``index`` position.

        The output shape is controlled by ``self.yields_events`` and
        ``self.yields_crops``: in the first case an entire event is returned,
        a single crop otherwise.

        Returns
        -------
        noisy: np.ndarray
            A single noisy example, of shape=(1,H,W).
        clear: np.ndarray
            A single clear example, of shape=(1,H,W) if ``self.has_target`` is
            ``True``, else ``None``.
        """
        if self.has_target:
            return self.noisy[index], self.clear[index]
        return self.noisy[index]
