# This file is part of DUNEdn by M. Rossi
from pathlib import Path
from datetime import datetime as dtm
from dunedn.configdn import get_dunedn_path
import shutil


def check(check_instance, check_list):
    if not check_instance in check_list:
        raise NotImplementedError("Operation not implemented")


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # configcard checks
        check(self.model, ["cnn", "gcnn", "uscg"])
        check(self.task, ["roi", "dn"])
        check(self.channel, ["induction", "collection"])

        # model parameters
        # self.a = 0.5 # balancing between loss function contributions
        # self.k = 8 # for cnn | gcnn model only
        # self.input_channels = 1 # for cnn | gcnn model only
        # self.hidden_channels = 32 # for cnn | gcnn model only

        # logs
        # self.epoch_log = 1
        # self.epoch_test_start = 0
        # self.epoch_test = 1

        # self.t = 0.5 # shouldn't need this

        self.load = False if (self.load_path is None) else True
        # self.load_epoch = 100

        # self.save = True

    def build_directories(self, output=None):
        """
        Saves in the attributes the output directory tree.

        Parameters
        ----------
            - output: Path, name of the output folder

        """
        if output is not None:
            output_ch = output / f"{self.channel}"
            if output_ch.is_dir():
                if self.force:
                    # remove the existing folder
                    print(f"WARNING: Overwriting {output_ch} directory with new model")
                    shutil.rmtree(output_ch)
                else:
                    print('Delete or run with "--force" to overwrite.')
                    exit(-1)
        else:
            date = dtm.now().strftime("%y%m%d_%H%M%S")
            output = get_dunedn_path().parent / f"output/{date}"

        self.dir_output = output / f"{self.channel}"

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
