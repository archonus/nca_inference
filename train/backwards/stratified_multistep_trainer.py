from typing import Generator, List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from ...utils import raise_if
from ...utils.visualisation import to_rgba
from .multistep_trainer import BackwardMultiStepTrainerBase


class BackwardsStratifiedMultiStepTrainer(BackwardMultiStepTrainerBase):
    """Trainer using multiple runs and the multi-step training strategy."""

    def __init__(
        self,
        runs: List[List[Tensor]],
        *,
        seed,
        learning_rate=2e-3,
        batch_size=8,
        log_dir="train_log",
        n_iter_range=32,
        n_iter_upper=96,
    ):
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            log_dir=log_dir,
            seed=seed,
        )
        raise_if(len(runs) < 1, "No runs provided")
        raise_if(
            batch_size % len(runs) != 0,
            "Batch size must be divisible by number of runs",
        )

        self.n_iter_upper = min(
            n_iter_upper, *[len(run) for run in runs]
        )  # Exclusive upper bound of iterations such that will stay in range for all runs
        raise_if(
            n_iter_range >= self.n_iter_upper,
            f"Range for the iterations must be less than upper bound of {self.n_iter_upper}",
        )
        self.n_iter_lower = self.n_iter_upper - n_iter_range

        self.runs: List[Tensor] = [tf.convert_to_tensor(run) for run in runs]
        self.n_runs = len(self.runs)
        self.run_lengths = np.array([tensor.shape[0] for tensor in self.runs])

        self.h, self.w, self.c = self.runs[0].shape[1:]

        self._dataset = tf.data.Dataset.from_generator(
            self._data_generator,
            output_signature=(
                tf.TensorSpec(
                    shape=[self.batch_size, self.h, self.w, self.c], dtype=tf.float32
                ),
                tf.TensorSpec(
                    shape=[self.batch_size, self.h, self.w, self.c], dtype=tf.float32
                ),
                tf.TensorSpec(shape=[], dtype=tf.int32),
            ),
        ).prefetch(tf.data.experimental.AUTOTUNE)

    def _data_generator(self) -> Generator[Tuple[Tensor, Tensor, int], None, None]:
        """
        Generate batches of data for training

        Yields:
            Tuple[Tensor, Tensor, int]: Tuple of the input data, output data and number of iterations to run
        """
        while True:
            iterations = tf.random.uniform(
                shape=[],
                minval=self.n_iter_lower,
                maxval=self.n_iter_upper,
                dtype=tf.int32,
            )  # Generate number of iterations to perform
            upper_bounds = (
                self.run_lengths[:, np.newaxis] - iterations
            )  # Of shape (r,1) to broadcast to correct shape in randint
            indices = np.random.randint(
                0,
                high=upper_bounds,
                size=(self.n_runs, self.batch_size // self.n_runs),
                dtype=np.int32,
            )  # Of shape (r,b/r)
            x0 = tf.concat(
                [tf.gather(self.runs[i], indices[i]) for i in range(self.n_runs)],
                axis=0,
            )  # Gather produces tensors of shape b/r, h, w, c. Concat to b, h, w, c
            y = tf.concat(
                [
                    tf.gather(self.runs[i], indices[i] + iterations)
                    for i in range(self.n_runs)
                ],
                axis=0,
            )
            yield x0, y, iterations

    @property
    def training_dataset(self) -> tf.data.Dataset:
        return self._dataset

    def get_training_state(self):
        state_dict = super().get_training_state()
        state_dict["n_iter_lower"] = self.n_iter_lower
        state_dict["n_iter_upper"] = self.n_iter_upper
        state_dict["run_lengths"] = self.run_lengths
        return state_dict


class BackwardsMultiRunTrainerRGBA(BackwardsStratifiedMultiStepTrainer):
    """Trainer using multiple runs and the multi-step training strategy with an RGBA loss function."""

    @tf.function
    def loss_function(self, x: Tensor, y: Tensor) -> Tensor:
        return tf.reduce_mean(tf.square(to_rgba(y) - to_rgba(x)))
