from typing import Tuple, Optional, Union

import numpy as np
import tensorflow as tf
from numpy.typing import ArrayLike
from tensorflow import Tensor

from ..nca.models import NCA
from ..utils import raise_if
from ..utils.image_loader import ImageLoader
from ..utils.visualisation import to_rgba
from .trainer import Trainer


class SamplePool:
    """
    Class representing a sample of values, potentially taken from a larger pool

    Attributes:
        x (ArrayLike): The values
        parent (SamplePool, optional): Parent `SamplePool` from which the values are drawn
        source_indices (ArrayLike, optional): Indices into the parent from which the sample is drawn
    """

    def __init__(
        self,
        x: ArrayLike,
        parent: Optional["SamplePool"] = None,
        source_indices: Optional[ArrayLike] = None,
    ):
        """
        Constructor for a `SamplePool` instance

        Args:
            x (ArrayLike): The values. Must be subscriptable by `ArrayLike` to support sampling
            parent (SamplePool, optional): Parent from which the values are drawn. Defaults to None.
            source_indices (ArrayLike, optional): Indices into the parent from which the sample is drawn.
                Should be none if and only if `parent` is None
                Defaults to None.

        Raises:
            ValueError: If `parent` is None and `source_indices` is not None or vice versa
        """
        self.x = x
        raise_if(
            (parent is None) != (source_indices is None),
            "Either both of parent and source_indices should be provided, or neither should be provided",
        )
        self.parent = parent
        self.source_indices = source_indices

    def sample(self, n: int) -> "SamplePool":
        """Generate new `SamplePool` of size `n`"""
        idx = np.random.choice(len(self.x), size=n, replace=False)  # Generate n indices
        batch = self.x[idx]  # Select those indices
        sample = SamplePool(batch, parent=self, source_indices=idx)
        return sample

    def write_back(self, x: ArrayLike):
        """
        Save the values back to the pool, and to parent if applicable

        Args:
            x (ArrayLike): Value to write back
        """
        self.x[:] = x
        if self.parent is not None and self.source_indices is not None:
            self.parent.x[self.source_indices] = (
                self.x
            )  # Write the changes back to the parent as well. This is not recursive.


class SamplePoolTrainer(Trainer):
    """Trainer for the NCA model using a sample pool of images"""

    def __init__(
        self,
        image_loader: ImageLoader,
        target_img: Tensor,
        learning_rate: float = 2e-3,
        pool_size=1024,
        batch_size=8,
        damage=3,
        log_dir="train_log",
    ):
        """
        Constructor for `SamplePoolTrainer` instances

        Args:
            image_loader (ImageLoader): ImageLoader instance to use for loading images
            target_img (Tensor): Target image to use for training. This is the image which the NCA will try to generate
            learning_rate (float, optional): Learning rate. Defaults to 2e-3.
            pool_size (int, optional): Size of the pool of images used in training. Defaults to 1024.
            batch_size (int, optional): Size of batches used at each update. Defaults to 8.
            damage (int, optional): Number of images to induce damage on. Defaults to 3.
            log_dir (str, optional): Directory for training output to be saved in.
        """
        super().__init__(
            learning_rate=learning_rate, batch_size=batch_size, log_dir=log_dir
        )
        self.image_loader = image_loader
        self.padded_target = image_loader.pad_target(target_img)
        self.h, self.w = self.padded_target.shape[:2]
        self.seed = image_loader.generate_seed(self.h, self.w)
        self.pool_size = pool_size
        self.pool = SamplePool(
            x=np.repeat(self.seed[None, ...], repeats=self.pool_size, axis=0)
        )
        self.damage_n = damage

    def loss_function(self, x: Union[ArrayLike, Tensor]) -> Tensor:
        """Compute the loss with respect to the target image for a batch of states

        Args:
            x (ArrayLike | Tensor): State grid of shape (b, h, w, c)

        Returns:
            Tensor: Loss vector of shape (b,)
        """
        return tf.reduce_mean(
            tf.square(to_rgba(x) - self.padded_target), [-2, -3, -1]
        )  # Returns vector of shape (b,)

    @tf.function
    def train_step(self, nca: NCA, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Train the model for a single step

        Args:
            nca (NCA): Model to train
            x (Tensor): Input states of shape (b, h, w, c)

        Returns:
            Tuple[Tensor, Tensor]: Tuple of the updated states and the loss
        """
        iterations = tf.random.uniform(shape=[], minval=64, maxval=96, dtype=tf.int32)
        with tf.GradientTape() as g:
            for _ in tf.range(iterations):
                x = nca(x)
            loss = tf.reduce_mean(self.loss_function(x))
        grads = g.gradient(loss, nca.trainable_weights)
        self.apply_gradients(nca, grads, self.optimizer)
        return x, loss

    def train_loop(self, nca: NCA, epochs=8000, start_epoch=0, pool: Optional[SamplePool] = None):
        self.pool = pool
        if self.pool is None:
            self.pool = SamplePool(
                x=np.repeat(self.seed[None, ...], repeats=self.pool_size, axis=0)
            )
        for epoch in range(start_epoch, epochs + 1):
            self.epoch = epoch
            batch = self.pool.sample(self.batch_size)
            x0 = batch.x
            loss_rank = self.loss_function(x0).numpy().argsort()[::-1]
            x0 = x0[loss_rank]  # Sort in descending order by the loss
            x0[:1] = self.seed  # Replace the highest loss with the seed
            if self.damage_n > 0:
                damage_mask = (
                    1.0
                    - ImageLoader.make_circle_masks(
                        self.h, self.w, n=self.damage_n
                    ).numpy()[..., np.newaxis]
                )
                x0[-self.damage_n :] *= damage_mask  # Damage the lowest loss samples

            x, loss = self.train_step(nca, tf.convert_to_tensor(x0))

            batch.write_back(x)  # Write back the new values to the sample pool

            self.losses.append(loss.numpy())
            self.output(epoch=epoch, loss=loss.numpy(), x0=x0, x=x, nca=nca)

    def get_training_state(self):
        state_dict = super().get_training_state()
        state_dict["pool_size"] = self.pool_size
        state_dict["damage_n"] = self.damage_n
        state_dict["h"] = self.h
        state_dict["w"] = self.w
        return state_dict
