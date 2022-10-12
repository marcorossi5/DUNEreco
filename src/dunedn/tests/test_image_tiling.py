"""
    Ensures DUNEdn croppiing utility runs.
"""
import numpy as np
from dunedn.networks.gcnn.gcnn_net_utils import Converter
from dunedn.geometry.helpers import evt2planes, planes2evt
from dunedn.geometry.pdune import geometry


def run_event_to_planes(events):
    iplanes, cplanes = evt2planes(events)
    assert len(iplanes.shape) == 4, f"got iplanes rank: {len(iplanes.shape)}"
    assert len(cplanes.shape) == 4, f"got cplanes rank: {len(cplanes.shape)}"
    assert iplanes.shape[1:] == (
        1,
        geometry["nb_ichannels"],
        geometry["nb_tdc_ticks"],
    ), f"got iplanes shape: {iplanes.shape[1:]}"
    assert cplanes.shape[1:] == (
        1,
        geometry["nb_cchannels"],
        geometry["nb_tdc_ticks"],
    ), f"got cplanes shape: {cplanes.shape[1:]}"
    reco_events = planes2evt(iplanes, cplanes)
    np.testing.assert_equal(events, reco_events)


def run_crop_converter(events):
    # test converter that exactly divide the event width
    converter1 = Converter((80, 1000))
    splits = converter1.image2crops(events)
    assert len(splits.shape) == 4, f"got splits rank: {len(splits.shape)}"
    assert splits.shape[1:] == (1, 80, 1000), f"got splits shape: {splits.shape[1:]}"
    reco_events = converter1.crops2image(splits)
    np.testing.assert_equal(events, reco_events)

    # test converter that does not exactly divide the event width
    converter2 = Converter((80, 512))
    splits = converter2.image2crops(events)
    assert len(splits.shape) == 4, f"got splits rank: {len(splits.shape)}"
    assert splits.shape[1:] == (1, 80, 512), f"got splits shape: {splits.shape[1:]}"
    reco_events = converter2.crops2image(splits)
    np.testing.assert_equal(events, reco_events)


def test_image_tiling():
    events = np.arange(2 * 15360 * 6000).reshape(2, 1, 15360, 6000) * 0.1
    run_event_to_planes(events)
    run_crop_converter(events)


if __name__ == "__main__":
    test_image_tiling()
