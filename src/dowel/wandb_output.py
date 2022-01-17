"""A `dowel.logger.LogOutput` for wandb.

It receives the input data stream from `dowel.logger`, then add them to the current wandb experiment

Note:
    WandB experiment has to be running before the logger gets started.

"""

import wandb
import functools

import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
# import tensorboardX as tbX
# try:
    # import tensorflow as tf
# except ImportError:
    # tf = None

from dowel import Histogram
from dowel import LoggerWarning
from dowel import LogOutput
from dowel import TabularInput
from dowel.utils import colorize


class WandBOutput(LogOutput):
    """WandB output for logger.

    """

    def __init__(self, additional_x_axes=None, x_axis=None,
                 histogram_samples=1e3):

        if x_axis is None:
            assert not additional_x_axes, (
                'You have to specify an x_axis if you want additional axes.')

        additional_x_axes = additional_x_axes or []

        self._x_axis = x_axis
        self._additional_x_axes = additional_x_axes
        self._histogram_samples = int(histogram_samples)

        self._waiting_for_dump = []
        self._default_step = 0

        self._warned_once = set()
        self._disable_warnings = False

    @property
    def types_accepted(self):
        """Return the types that the logger may pass to this output."""
        return (TabularInput, )

    def record(self, data, prefix=''):
        """Add data to wandb experimet.

        Args:
            data: The data to be logged by the output.
            prefix(str): A prefix placed before a log entry in text outputs.

        """

        if isinstance(data, TabularInput):
            self._waiting_for_dump.append(
                functools.partial(self._record_tabular, data))
        else:
            raise ValueError('Unacceptable type.')

        pass

    def _record_tabular(self, data, step):
        if self._x_axis:
            nonexist_axes = []
            for axis in [self._x_axis] + self._additional_x_axes:
                if axis not in data.as_dict:
                    nonexist_axes.append(axis)
            if nonexist_axes:
                self._warn('{} {} exist in the tabular data.'.format(
                    ', '.join(nonexist_axes),
                    'do not' if len(nonexist_axes) > 1 else 'does not'))

        for key, value in data.as_dict.items():
            if isinstance(value, np.ScalarType) and self._x_axis in data.as_dict:
                if self._x_axis is not key:
                    x = data.as_dict[self._x_axis]
                    self._record_kv(key, value, x)

                for axis in self._additional_x_axes:
                    if key is not axis and key in data.as_dict:
                        x = data.as_dict[axis]
                        self._record_kv('{}/{}'.format(key, axis), value, x)
            else:
                self._record_kv(key, value, step)
            data.mark(key)


    def _record_kv(self, key, value, step):
        if isinstance(value, np.ScalarType):
            wandb.log({key: value}, step=step)
        elif isinstance(value, plt.Figure):
            wandb.log({key: value}, step=step)
        elif isinstance(value, scipy.stats._distn_infrastructure.rv_frozen):
            shape = (self._histogram_samples, ) + value.mean().shape
            hist = np.histogram(value.rvs(shape))
            wandb_hist = wandb.Histogram(hist)
            wandb.log({key: wandb_hist}, step=step)
        elif isinstance(value, scipy.stats._multivariate.multi_rv_frozen):
            hist = np.histogram(value.rvs(self._histogram_samples))
            wandb_hist = wandb.Histogram(hist)
            wandb.log({key: wandb_hist}, step=step)
        if isinstance(value, np.ndarray):
            # If a numpy array is supplied we assume the dimensions are, in order: time, channels, width, height
            wandb.log( {key: wandb.Video(value, fps=60, format="mp4")})
        elif isinstance(value, Histogram):
            hist = np.histogram(value)
            wandb_hist = wandb.Histogram(hist)
            wandb.log({key: wandb_hist}, step=step)




    def dump(self, step=None):
        """Dump the contents of this output.

        :param step: The current run step.
        """

        # Log the tabular inputs, now that we have a step
        for p in self._waiting_for_dump:
            p(step or self._default_step)
        self._waiting_for_dump.clear()

        # Flush output files
        # for w in self._writer.all_writers.values():
            # w.flush()

        self._default_step += 1

    def close(self):
        """Close any files used by the output."""
        pass


    def _warn(self, msg):
        """Warns the user using warnings.warn.

        The stacklevel parameter needs to be 3 to ensure the call to logger.log
        is the one printed.
        """
        if not self._disable_warnings and msg not in self._warned_once:
            warnings.warn(colorize(msg, 'yellow'),
                          NonexistentAxesWarning,
                          stacklevel=3)
        self._warned_once.add(msg)
        return msg


class NonexistentAxesWarning(LoggerWarning):
    """Raise when the specified x axes do not exist in the tabular."""
