# This file is part of DUNEdn by M. Rossi
"""
    This module contains the Args class, that keeps track of all runtime settings.
"""
from pathlib import Path
from datetime import datetime as dtm
from dunedn.configdn import get_dunedn_path
from dunedn.utils.utils import check
import shutil


class Args:
    """ Class that tracks all the needed runtime settings."""

    def __init__(self, **kwargs):
        """
        Updates attributes from kwargs.

        Parameters
        ----------
            - kwargs: dict, key-value pairs to be stored as object attributes
        """
        self.__dict__.update(kwargs)

        # configcard checks
        check(self.model, ["cnn", "gcnn", "uscg"])
        check(self.task, ["roi", "dn"])
        check(self.channel, ["induction", "collection"])

        self.dataset_dir = Path(self.dataset_dir)
        self.crop_size = (self.crop_edge,) * 2

        self.load = False if (self.load_path is None) else True

    def build_directories(self, output=None):
        """
        Builds the output directory tree to store training results and logs.

        Parameters
        ----------
            - output: Path, name of the output folder. If None, generate a
                      unique output directory based on the current date and time.

        """
        if self.output is not None:
            output = self.output / f"{self.channel}"
            if output.is_dir():
                if self.force:
                    print(f"WARNING: Overwriting {output} directory with new model")
                    shutil.rmtree(output)
                else:
                    print('Delete or run with "--force" to overwrite.')
                    exit(-1)
            else:
                print(f"[+] Creating output directory at {output}")
        else:
            date = dtm.now().strftime("%y%m%d_%H%M%S")
            output = get_dunedn_path().parent / f"output/{date}/{self.channel}"
            print(f"[+] Creating output directory at {output}")

        self.dir_output = output

        self.dir_timings = self.dir_output / "timings"
        self.dir_timings.mkdir(parents=True, exist_ok=True)

        self.dir_testing = self.dir_output / "testing"
        self.dir_testing.mkdir(exist_ok=True)

        self.dir_final_test = self.dir_output / "final_test"
        self.dir_final_test.mkdir(exist_ok=True)

        self.dir_metrics = self.dir_output / "metrics"
        self.dir_metrics.mkdir(exist_ok=True)

        self.dir_saved_models = self.dir_output / "saved_models"
        self.dir_saved_models.mkdir(exist_ok=True)
