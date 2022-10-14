"""
    This module makes a gcnn inference pass.

    .. code-block:: text

        $ python examples/gcnn_inference.py \
            -c ../output/cnn/models/cnn.pth \
            -r runcards/default.yaml \
            -e ../test_dataset/val/evts/rawdigit_evt0.npy \
            -d cuda:0
"""

import argparse
from pathlib import Path
import pprint
from dunedn.networks import TilingDataset, load_and_compile_gcnn_network
from dunedn.utils import load_runcard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runcard", type=Path, help="The runcard file path", dest="runcard_path")
    parser.add_argument("-e", "--event", type=Path, dest="event_path")
    parser.add_argument("-d", "--device", help="The device hosting the computation", dest="dev")
    args = parser.parse_args()

    modeltype = "cnn"
    ckpt_path = Path("../../output/tmp/models/cnn/cnn.pth")
    setup = load_runcard(args.runcard_path)
    msetup = setup["model"][modeltype]
    pprint.pprint(msetup, indent=2)

    event = TilingDataset(args.event_path, msetup["batch_size"], msetup["crop_size"], has_target=True)

    network = load_and_compile_gcnn_network(msetup, checkpoint_filepath=ckpt_path)
    y_pred, logs = network.predict(event, dev=args.dev)

    pprint.pprint(logs, indent=2)

if __name__ == "__main__":
    main()
