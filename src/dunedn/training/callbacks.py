"""This module implements onjects to keep track of training details. """
import logging
from pprint import pformat
from pathlib import Path
from typing import List
import numpy as np
import torch
from dunedn import PACKAGE

logger = logging.getLogger(PACKAGE + ".train")


class Callback:
    def __init__(self):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_eval_begin(self, logs=None):
        pass

    def on_eval_end(self, logs=None):
        pass

    def set_model(self, model):
        self.model = model


class CallbackList(Callback):
    def __init__(self, callbacks: List[Callback], model=None):
        self.callbacks = callbacks

        if model:
            self.set_model(model)

    def set_model(self, model):
        self.model = model
        for callback in self.callbacks:
            callback.set_model(model)

    def hook(self, hook_name: str, logs: dict):
        """An utility function to call each callback method.

        Parameters
        ----------
        hook_name: str
            The name of the method to be called.
        logs: dict
            The dictionary to be logged.
        """
        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            hook(logs)

    def batch_hook(self, hook_name: str, batch: int, logs: dict):
        """An utility function to call each callback method.

        Parameters
        ----------
        hook_name: str
            The name of the method to be called.
        batch: int
            The current batch number.
        logs: dict
            The dictionary to be logged.
        """
        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            hook(batch, logs)

    def epoch_hook(self, hook_name: str, epoch: int, logs: dict):
        """An utility function to call each callback method.

        Parameters
        ----------
        hook_name: str
            The name of the method to be called.
        epoch: int
            The current epoch number.
        logs: dict
            The dictionary to be logged.
        """
        for callback in self.callbacks:
            hook = getattr(callback, hook_name)
            hook(epoch, logs)

    def on_train_begin(self, logs=None):
        self.hook("on_train_begin", logs)

    def on_train_end(self, logs=None):
        self.hook("on_train_end", logs)

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_hook("on_train_batch_begin", batch, logs)

    def on_train_batch_end(self, batch, logs=None):
        self.batch_hook("on_train_batch_end", batch, logs)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_hook("on_epoch_begin", epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_hook("on_epoch_end", epoch, logs)

    def on_eval_begin(self, logs=None):
        self.hook("on_eval_begin", logs)

    def on_eval_end(self, logs=None):
        self.hook("on_eval_end", logs)


class History(Callback):
    def __init__(self):
        super().__init__()
        self.history = {}

    def on_train_begin(self, logs=None):
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        # Set the history attribute on the model after the epoch ends. This will
        # make sure that the state which is set is the latest one.
        self.model.history = self


class ModelCheckpoint(Callback):
    def __init__(
        self,
        file_path: Path,
        monitor: str = "val_loss",
        verbose: int = 0,
        save_best_only: bool = False,
        mode: str = "auto",
        initial_value_threshold: float = None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(file_path, Path):
            file_path = file_path.as_posix()
        self.file_path: str = file_path
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.mode = mode
        self.best = initial_value_threshold

        self.epochs_since_last_save = 0
        self.save_freq = "epoch"

        if "period" in kwargs:
            self.period = kwargs["period"]
            logging.warning(
                "`period` argument is deprecated. Please use `save_freq` "
                "to specify the frequency in number of batches seen."
            )
        else:
            self.period = 1

        if mode not in ["auto", "min", "max"]:
            logger.warning(
                "ModelCheckpoint mode %s is unknown, " "fallback to auto mode.",
                mode,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
            if self.best is None:
                self.best = np.Inf
        elif mode == "max":
            self.monitor_op = np.greater
            if self.best is None:
                self.best = -np.Inf
        else:
            if "acc" in self.monitor or self.monitor.startswith("fmeasure"):
                self.monitor_op = np.greater
                if self.best is None:
                    self.best = -np.Inf
            else:
                self.monitor_op = np.less
                if self.best is None:
                    self.best = np.Inf

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1

        if self.save_freq == "epoch":
            self._save_model(epoch=epoch, batch=None, logs=logs)

    def _save_model(self, epoch: int, batch: int, logs: dict):
        """Saves the model.

        Parameters
        ----------
        epoch: int
            The epoch this iteration is in.
        batch: int
            The batch this iteration is in. `None` if the `save_freq` is set to
            `epoch`.
        logs: the `logs` dict passed in to `on_epoch_end`.
        """
        logs = logs or {}

        if (
            isinstance(self.save_freq, int)
            or self.epochs_since_last_save >= self.period
        ):
            # Block only when saving interval is reached.
            self.epochs_since_last_save = 0
            file_path = self._get_file_path(epoch, batch, logs)

            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logger.warning(
                        "Can save best model only with %s available, " "skipping.",
                        self.monitor,
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            logger.info(
                                f"\nEpoch {epoch + 1}: {self.monitor} "
                                "improved "
                                f"from {self.best:.5f} to {current:.5f}, "
                                f"saving model to {file_path}"
                            )
                        self.best = current
                        # put the saving here
                        net = (
                            self.model.module
                            if isinstance(self.model, torch.nn.DataParallel)
                            else self.model
                        )
                        torch.save(net.state_dict(), file_path)
                    else:
                        if self.verbose > 0:
                            logger.info(
                                f"\nEpoch {epoch + 1}: "
                                f"{self.monitor} did not improve "
                                f"from {self.best:.5f}"
                            )
            else:
                if self.verbose > 0:
                    logger.info(f"\nEpoch {epoch + 1}: saving model to {file_path}")
                net = (
                    self.model.module
                    if isinstance(self.model, torch.nn.DataParallel)
                    else self.model
                )
                torch.save(net.state_dict(), file_path)

    def _get_file_path(self, epoch, batch, logs):
        """Returns the file path for checkpoint."""

        try:
            # `file_path` may contain placeholders such as
            # `{epoch:02d}`,`{batch:02d}` and `{mape:.2f}`. A mismatch between
            # logged metrics and the path's placeholders can cause formatting to
            # fail.
            if batch is None or "batch" in logs:
                file_path = self.file_path.format(epoch=epoch + 1, **logs)
            else:
                file_path = self.file_path.format(
                    epoch=epoch + 1, batch=batch + 1, **logs
                )
        except KeyError as e:
            raise KeyError(
                f'Failed to format this callback file_path: "{self.file_path}". '
                f"Reason: {e}"
            )
        # TODO: distributed utils may be implemented. The only worker to save
        # the file is the chief. Others just fill a temporary directory that
        # should be removed at the end of the training
        self._write_file_path = file_path
        return self._write_file_path
