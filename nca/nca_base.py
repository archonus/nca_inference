from abc import ABC, abstractmethod
from typing import Callable, List, Union

import numpy as np
import tensorflow as tf
from tensorflow import Tensor, keras

from ..utils import pickle_save, raise_if


class NCA(keras.Model, ABC):
    """Abstract base class for Neural Cellular Automata models"""

    def __init__(self, n_channels, live_threshold, fire_rate):
        """
        Constructor for building an instance of an NCA model

        Args:
            n_channels (int, optional): Number of channels of the input.
            live_threshold (float, optional): The threshold for determining whether a cell is alive. If the value of the alpha channel is greater than the threshold, then the cell is alive; if the cell has no living neighbours, then it is dead.
            fire_rate (float, optional): The dropout rate for the cell updates.
        """
        keras.Model.__init__(self)
        self.n_channels = n_channels
        self.live_threshold = live_threshold
        self.fire_rate = fire_rate
        self.perception_layer = self.construct_perception_layer()
        self.model = self.construct_update_model()

        self(tf.zeros([1, 3, 3, self.n_channels]))  # Dummy call to build the model

    @abstractmethod
    def construct_update_model(self) -> keras.Model:
        """
        Construct the `keras.Model` object which determines the update rule of the NCA

        Returns:
            keras.Model: The model which takes in a perception vector of size `(..., n_kernels * n_channels)` and produces the update to the cell(s) of shape `(..., n_channels)`
        """

    @abstractmethod
    def construct_perception_layer(
        self,
    ) -> Union[keras.layers.Layer, Callable[[Tensor], Tensor]]:
        """
        Construct the perception layer which takes in a state grid of shape `(..., n_channels)` and produces a perception vector of shape `(..., n_kernels * n_channels)`

        Returns:
            Union[Callable[keras.layers.Layer, [Tensor], Tensor]]: Perception layer or callable which will generate the perception vector from the state grid
        """

    @abstractmethod
    def get_living_mask(self, x: Tensor) -> Tensor:
        """
        Compute the living mask of the state grid

        Args:
            x (Tensor): State grid of shape ([b], h, w, c)

        Returns:
            Tensor: Boolean tensor of same shape as `x` specifying the non-dead cells (alive or growing)
        """

    @tf.function
    def call(self, x: Tensor) -> Tensor:
        """
        Update the state grid based on the update rule

        Args:
            x (Tensor): State grid of shape ([b], h, w, `n_channels`)

        Returns:
            Tensor: The updated state grid of shape ([b], h, w, `n_channels`)
        """

        pre_live_mask = self.get_living_mask(x)
        perception_vector = self.perception_layer(x)
        dx = self.model(perception_vector)
        update_mask = tf.random.uniform(tf.shape(x[..., :1])) <= self.fire_rate
        x += dx * tf.cast(update_mask, tf.float32)
        post_live_mask = self.get_living_mask(x)
        live_mask = pre_live_mask & post_live_mask
        return x * tf.cast(live_mask, tf.float32)

    def run(self, x0: Tensor, n=500) -> List[Tensor]:
        """Run model for `n` iterations on initial state grid `x0`

        Args:
            x0 (Tensor): Initial state, of shape (h, w, c).
                Note: this function does not operate on batches
            n (int, optional): Number of iterations to run. Defaults to 500.

        Returns:
            List[Tensor]: Frames of each time step, each tensor being of shape (h, w, c)
        """
        if len(x0.shape) == 4:  # Has batch dimension
            raise_if(x0.shape[0] > 1, "Cannot operate on batches")
            x0 = x0[0]  # Strip the batch dimension
        frames = [x0]
        for _ in range(n):
            x = frames[-1][np.newaxis, ...]  # Add batch dimension
            x = self(x)[0]  # Discard batch dimension
            frames.append(x)
        return frames

    def save_model_weights(self, out_file_name: str):
        """Saves the model's weights as a flat list of NumPy arrays using pickle

        Args:
            out_file_name (str): File name to save to
        """
        pickle_save(self.get_weights(), out_file_name)
