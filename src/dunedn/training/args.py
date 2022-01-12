# This file is part of DUNEdn by M. Rossi
"""
    This module contains the Args class, that keeps track of all runtime settings.
"""
import logging
from pathlib import Path
from datetime import datetime as dtm
import shutil
from dunedn.configdn import PACKAGE
from dunedn.configdn import get_output_path
from dunedn.utils.utils import check

# instantiate logger
logger = logging.getLogger(PACKAGE)


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

        self.load = self.load_path is not None

        # build directories
        self.dir_output = None
        self.dir_timings = None
        self.dir_testing = None
        self.dir_final_test = None
        self.dir_metrics = None
        self.dir_saved_models = None

    def build_directories(self):
        """
        Builds the output directory tree to store training results and logs.
        """
        if self.output is not None:
            output = self.output / f"{self.channel}"
            if output.is_dir():
                if self.force:
                    logger.warning(
                        "Overwriting %s directory with new model", output.as_posix()
                    )
                    shutil.rmtree(output)
                else:
                    logger.critical('Delete or run with "--force" to overwrite.')
                    exit(-1)
            else:
                logger.info("Creating output directory at %s", output.as_posix())
        else:
            date = dtm.now().strftime("%y%m%d_%H%M%S")
            output = get_output_path() / f"{date}/{self.channel}"
            logger.info("Creating output directory at %s", output.as_posix())

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
