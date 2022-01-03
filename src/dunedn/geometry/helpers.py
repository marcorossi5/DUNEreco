# This file is part of DUNEdn by M. Rossi
"""This module contains the geometry helper functions."""

import numpy as np
from dunedn.geometry.pdune import (
    nb_tdc_ticks,
    nb_ichannels,
    nb_apas,
    nb_apa_channels,
)


def evt2planes(event):
    """
    Convert planes to event
    Input:
        event: array-like array
            inputs of shape (nb_event_channels, nb_tdc_ticks)
    Output: np.array
        induction and collection arrays of shape type (N,C,H,W)
    """
    base = np.arange(nb_apas).reshape(-1, 1) * nb_apa_channels
    iidxs = [[0, nb_ichannels, 2 * nb_ichannels]] + base
    cidxs = [[2 * nb_ichannels, nb_apa_channels]] + base
    inductions = []
    for start, idx, end in iidxs:
        induction = [event[start:idx], event[idx:end]]
        inductions.extend(induction)
    collections = []
    for start, end in cidxs:
        collections.append(event[start:end])
    return np.stack(inductions)[:, None], np.stack(collections)[:, None]


def planes2evt(inductions, collections):
    """
    Convert planes to event
    Input:
        inductions, collections: array-like
            inputs of shape type (N,C,H,W)
    Output: np.array
        event array of shape (nb_event_channels, nb_tdc_ticks)
    """
    inductions = np.array(inductions).reshape(-1, 2 * nb_ichannels, nb_tdc_ticks)
    collections = np.array(collections)[:, 0]
    event = []
    for i, c in zip(inductions, collections):
        event.extend([i, c])
    return np.concatenate(event)
