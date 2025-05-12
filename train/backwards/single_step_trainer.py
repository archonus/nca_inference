from typing import List, Tuple, Optional

import tensorflow as tf
from tensorflow import Tensor

from ...nca import NCA
from ..trainer import Trainer


class BackwardsSingleStepTrainer(Trainer):
    """Trainer for training NCA models using the backwards single step method."""

    def __init__(
        self,
        train_xy: List[Tuple[Tensor, Tensor]],
        test_xy=None,
        *,
        seed=None,
        learning_rate=2e-3,
        batch_size=64,
        log_dir="train_log",
        reshuffle=False
    ):
        super().__init__(
            learning_rate=learning_rate,
            batch_size=batch_size,
            log_dir=log_dir,
            seed=seed,
        )
        self.train_x, self.train_y = zip(*train_xy)  # Zip is its own inverse
        self.train_x = list(self.train_x)
        self.train_y = list(self.train_y)  # Convert to lists
        self.test_xy = test_xy
        self.train_dataset = (
            tf.data.Dataset.from_tensor_slices((self.train_x, self.train_y))
            .shuffle(buffer_size=len(self.train_x), reshuffle_each_iteration=reshuffle)
            .batch(batch_size)
        )
        self.num_batches = len(self.train_dataset)

    @tf.function
    def train_step(self, nca: NCA, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Single step of training

        Args:
            nca (NCA): Model to be trained
            x (Tensor): Input training data, of shape (b,h,w,c)
            y (Tensor): Output training data of same shape as x
        Returns:
            Tuple[Tensor, Tensor]: Tuple of the updated states and the loss
        """
        with tf.GradientTape() as g:
            y_hat = nca(x)
            loss: Tensor = tf.reduce_mean(tf.square(y - y_hat))
        grads = g.gradient(loss, nca.trainable_weights)
        self.apply_gradients(nca, grads, self.optimizer)
        return y_hat, loss

    def train_loop(
        self,
        nca: NCA,
        epochs=8000,
        start_epoch=0,
        dataset: Optional[tf.data.Dataset] = None,
        n_show=8,
        **kwargs
    ):
        if dataset is not None:
            self.train_dataset = dataset
        n_show = min(n_show, self.batch_size)

        for epoch in tf.range(start_epoch, epochs + 1):
            self.epoch = epoch
            total_loss = 0
            for x_batch, y_batch in self.train_dataset:
                y_hat, batch_loss = self.train_step(nca, x_batch, y_batch)
                total_loss += batch_loss.numpy()
            self.losses.append(total_loss)
            self.output(
                epoch=epoch,
                nca=nca,
                loss=total_loss,
                x0=x_batch[:n_show],
                x=y_hat[:n_show],
                **kwargs
            )
