from abc import ABC, abstractmethod
from typing import Generator, Tuple, List, Union

import tensorflow as tf
from tensorflow import Tensor

from ...nca import NCA
from ...utils import raise_if
from ..trainer import Trainer


class BackwardMultiStepTrainerBase(Trainer, ABC):
    """Base class for training NCA models using the backwards multi-step training strategy."""

    def __init__(self, *, learning_rate, batch_size, log_dir, seed):
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            log_dir=log_dir,
            seed=seed,
        )
        self.batch_ns = []  # List of number of iterations ran for each batch

    @property
    @abstractmethod
    def training_dataset(self) -> tf.data.Dataset:
        """Training dataset, with each element consisting of (x,y,n) where x is the input, y is the output and n is the number of iterations"""

    @tf.function
    def loss_function(self, x: Tensor, y: Tensor) -> Tensor:
        """Loss function to be used for training"""
        return tf.reduce_mean(tf.square(y - x))

    @tf.function
    def train_step(
        self, nca: NCA, x: Tensor, y: Tensor, n: int
    ) -> Tuple[Tensor, Tensor]:
        """
        Single step of training

        Args:
            nca (NCA): Model to be trained
            x (Tensor): Input training data, of shape (b,h,w,c)
            y (Tensor): Output training data of same shape as x
            n (int): Number of iterations to run
        Returns:
            Tuple[Tensor, Tensor]: Tuple of the updated states and the loss
        """
        with tf.GradientTape() as g:
            for _ in tf.range(n):
                x = nca(x)
            loss: Tensor = self.loss_function(x, y)
        grads = g.gradient(loss, nca.trainable_weights)
        self.apply_gradients(nca, grads, self.optimizer)
        return x, loss

    def train_loop(self, nca: NCA, epochs=8000, start_epoch=0, **kwargs):
        n_epochs = epochs - start_epoch

        for i, (x0, y, n) in enumerate(self.training_dataset.take(n_epochs + 1)):
            self.epoch = i + start_epoch
            x, loss = self.train_step(nca, x0, y, n)

            self.losses.append(loss.numpy())
            self.batch_ns.append(n)
            self.output(
                nca=nca, epoch=self.epoch, loss=loss, x0=x0, x=x, iterations=n, **kwargs
            )

    def get_training_state(self):
        state_dict = super().get_training_state()
        state_dict["batch_ns"] = self.batch_ns
        return state_dict


class BackwardsMultiStepTrainer(BackwardMultiStepTrainerBase):
    """Trainer for the backwards multi-step training strategy."""

    def __init__(
        self,
        x: Union[List[Tensor], Tensor],
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
        self.x_frames = tf.stack(x) if isinstance(x, list) else x
        self.N, self.h, self.w, self.c = self.x_frames.shape  # Shape of the frames
        self.n_iter_upper = min(
            n_iter_upper, len(self.x_frames)
        )  # Exclusive upper bound for the number of iterations to run
        self.n_iter_lower = (
            self.n_iter_upper - n_iter_range
        )  # Inclusive lower bound for the number of iterations to run
        raise_if(
            n_iter_range >= self.n_iter_upper,
            f"Range for the iterations must be less than upper bound of {self.n_iter_upper}",
        )

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

    @property
    def training_dataset(self) -> tf.data.Dataset:
        return self._dataset

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
            upper_bound = len(self.x_frames) - iterations
            indices = tf.random.uniform(
                shape=[self.batch_size], minval=0, maxval=upper_bound, dtype=tf.int32
            )  # Generate indices
            x0 = tf.gather(self.x_frames, indices)
            y = tf.gather(self.x_frames, indices + iterations)  # Extract targets
            yield x0, y, iterations

    def get_training_state(self):
        state_dict = super().get_training_state()
        state_dict["n_iter_lower"] = self.n_iter_lower
        state_dict["n_iter_upper"] = self.n_iter_upper
        state_dict["n_frames"] = self.N
        return state_dict
