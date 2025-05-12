from typing import List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras

from ..utils import raise_if


class PerceptionLayer(tf.keras.layers.Layer):
    """Layer to generate perception vector from image using convolutional filters"""

    sobel_x_filter = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
    sobel_y_filter = sobel_x_filter.T
    identify_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    averaging_filter = np.ones((3, 3)) / 9.0

    def __init__(
        self,
        perception_filters: Optional[List[Union[Tensor, ArrayLike]]] = None,
        size: int = 3,
        n_channels: int = 16,
    ):
        """
        Initialize the PerceptionLayer with the given filters
        Args:
            perception_filters (List[Tensor], optional): List of filters to use. Defaults to [identify_filter, sobel_x_filter, sobel_y_filter].
            size (int, optional): Size of the filters. Defaults to 3.
                Each filter in `perception_filters` should be of shape (size, size).
        """
        super().__init__()
        self.size = size
        filters = []
        if perception_filters is None:
            perception_filters = [
                self.identify_filter,
                self.sobel_x_filter,
                self.sobel_y_filter,
                self.averaging_filter,
            ]
        for f in perception_filters:
            raise_if(
                f.shape[0] != size or f.shape[1] != size,
                f"Filter shape {f.shape} does not match size {size}",
                ValueError,
            )
            filters.append(tf.constant(f, dtype=tf.float32))

        self.filters = tf.stack(filters, -1)[:, :, None, :]  # Shape (size, size, 1, n_filters)
        self.perception_filter = tf.repeat(self.filters, n_channels, axis=2)

    def call(self, inputs: Tensor) -> Tensor:
        """
        Apply the perception filters to the input tensor

        Args:
            x (Tensor): Input tensor of shape (..., height, width, channels)

        Returns:
            Tensor: Output tensor of shape (..., height, width, num_filters * channels)
        """
        return tf.nn.depthwise_conv2d(
            inputs, filter=self.perception_filter, strides=[1, 1, 1, 1], padding="SAME"
        )


class LearnablePerceptionLayer(PerceptionLayer):
    """Layer to generate perception vector from image using learnable convolutional filters"""

    def __init__(self, n_filters=3, size=3, n_channels=16):
        keras.layers.Layer.__init__(self)
        self.perception_filter = self.add_weight(
            name="perception_filter",
            shape=(size, size, n_channels, n_filters),
            dtype=tf.float32,
            trainable=True,
        )
