from abc import ABC
from tensorflow import keras

from .perception_layer import PerceptionLayer, LearnablePerceptionLayer
from .nca_base import NCA


class AveragingPerceptionMixin(ABC):
    """
    Mixin class for NCA models which use an averaging filter in the perception filter.
    """

    def construct_perception_layer(self: NCA) -> keras.layers.Layer:
        return PerceptionLayer(
            perception_filters=[PerceptionLayer.identify_filter, PerceptionLayer.averaging_filter],
            n_channels=self.n_channels,
        )


class AveragingSobelPerceptionMixin(ABC):
    """
    Mixin class for NCA models which use an averaging and Sobel filters in the perception filter.
    """

    def construct_perception_layer(self: NCA) -> keras.layers.Layer:
        return PerceptionLayer(
            perception_filters=[
                PerceptionLayer.identify_filter,
                PerceptionLayer.averaging_filter,
                PerceptionLayer.sobel_x_filter,
                PerceptionLayer.sobel_y_filter,
            ],
            n_channels=self.n_channels,
        )


class LearnablePerceptionMixin(ABC):
    """
    Mixin class for NCA models which have a learnable kernel
    """

    def construct_perception_layer(self: NCA) -> keras.layers.Layer:
        return LearnablePerceptionLayer(
            n_filters=self.n_filters, size=3, n_channels=self.n_channels
        )
