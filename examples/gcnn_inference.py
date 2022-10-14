"""
    This module makes a gcnn inference pass.

    .. code-block:: text

        $ python examples/gcnn_inference.py \
            -r runcards/default.yaml \
            -e ../test_dataset/val/evts/rawdigit_evt0.npy \
            -d cuda:0
"""

import argparse
from pathlib import Path
import pprint
import numpy as np
from dunedn.networks import TilingDataset, load_and_compile_gcnn_network
from dunedn.utils import load_runcard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=Path, help="The model checkpoint file path", dest="ckpt_path")
    parser.add_argument("-r", "--runcard", type=Path, help="The runcard file path", dest="runcard_path")
    parser.add_argument("-e", "--event", type=Path, dest="event_path")
    parser.add_argument("-d", "--device", help="The device hosting the computation", dest="dev")
    args = parser.parse_args()

    modeltype = "cnn"
    setup = load_runcard(args.runcard_path)
    msetup = setup["model"][modeltype]
    pprint.pprint(msetup, indent=2)

    event = TilingDataset(args.event_path, msetup["batch_size"], msetup["crop_size"], has_target=True)

    network = load_and_compile_gcnn_network(msetup, checkpoint_filepath=args.ckpt_path)
    y_pred, logs = network.predict(event, dev=args.dev)

    pprint.pprint(logs, indent=2)

    out_path = args.ckpt_path.parent / f"{args.event_path.stem}_gcnn_dn{args.event_path.suffix}"
    np.save(out_path, y_pred[0,0])

if __name__ == "__main__":
    main()
