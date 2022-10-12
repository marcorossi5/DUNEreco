"""
    This module contains the geometry helper functions that transform events into
    planes and vice versa.
"""
from typing import Tuple
import numpy as np
from .pdune import geometry as pdune_geometry


def evt2planes(event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts event array to planes.

    Parameters
    ----------
    event: np.array
        Raw Digit array, of shape=(N,1,nb_event_channels, nb_tdc_ticks).

    Returns
    -------
    inductions: np.array
        Induction planes array, of shape=(N,C,H,W).
    collections: np.array
        Collection planes array, of shape=(N,C,H,W).
    """
    idxs = np.cumsum(
        [2 * pdune_geometry["nb_ichannels"], pdune_geometry["nb_cchannels"]]
    )
    base = (
        np.arange(pdune_geometry["nb_apas"]).reshape(-1, 1)
        * pdune_geometry["nb_apa_channels"]
    )
    split_idxs = (base + idxs[None]).flatten()[:-1]
    splits = np.split(event, split_idxs, axis=2)
    ishape = (-1, 1, pdune_geometry["nb_ichannels"], pdune_geometry["nb_tdc_ticks"])
    iplanes = np.stack(splits[::2], axis=1).reshape(ishape)
    cshape = (-1, 1, pdune_geometry["nb_cchannels"], pdune_geometry["nb_tdc_ticks"])
    cplanes = np.stack(splits[1::2], axis=1).reshape(cshape)
    return iplanes, cplanes


def planes2evt(inductions: np.ndarray, collections: np.ndarray) -> np.ndarray:
    """
    Converts planes back to event.

    Parameters
    ----------
    inductions: np.array
        Induction planes, of shape=(N,C,H,W).
    collections: np.array
        Collection planes, of shape=(N,C,H,W).

    Returns
    -------
    np.array
        Raw Digits array, of shape=(N, 1, nb_event_channels, nb_tdc_ticks).
    """
    nb_channels = inductions.shape[1]
    ishape = (
        -1,
        6,
        nb_channels,
        2 * pdune_geometry["nb_ichannels"],
        pdune_geometry["nb_tdc_ticks"],
    )
    inductions = inductions.reshape(ishape)
    cshape = (
        -1,
        6,
        nb_channels,
        pdune_geometry["nb_cchannels"],
        pdune_geometry["nb_tdc_ticks"],
    )
    collections = collections.reshape(cshape)

    # concatenate
    events = np.concatenate([inductions, collections], axis=3)
    nb_events = events.shape[0]

    # collapse 1,2 axes
    ev_shape = (nb_events, nb_channels, -1, pdune_geometry["nb_tdc_ticks"])
    events = events.transpose(0, 2, 1, 3, 4).reshape(ev_shape)
    return events
