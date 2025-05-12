import tensorflow as tf
from tensorflow import keras

from .nca_base import NCA
from .perception_layer import PerceptionLayer
from .perception_mixin import (
    AveragingPerceptionMixin,
    AveragingSobelPerceptionMixin,
    LearnablePerceptionMixin,
)


class MorphogenesisNCA(NCA):
    """NCA model architecture described by Mordvintsev et al. (2020)"""

    def __init__(
        self, n_channels=16, live_threshold=0.1, fire_rate=0.5, hidden_dim=128
    ):

        self.hidden_dim = hidden_dim
        super().__init__(
            n_channels=n_channels, live_threshold=live_threshold, fire_rate=fire_rate
        )

    def construct_update_model(self):

        return keras.Sequential(
            [
                keras.layers.Conv2D(self.hidden_dim, 1, activation=tf.nn.relu),
                keras.layers.Conv2D(
                    self.n_channels,
                    1,
                    activation=None,
                    kernel_initializer="zeros",
                ),
            ]
        )

    def construct_perception_layer(self):

        return PerceptionLayer(
            perception_filters=[
                PerceptionLayer.identify_filter,
                PerceptionLayer.sobel_x_filter,
                PerceptionLayer.sobel_y_filter,
            ],
            size=3,
            n_channels=self.n_channels,
        )

    def get_living_mask(self, x):
        alpha = x[:, :, :, 3:4]
        return (
            tf.nn.max_pool2d(alpha, ksize=3, strides=[1, 1, 1, 1], padding="SAME")
            >= self.live_threshold
        )


class TwoLayerNCA(MorphogenesisNCA):
    """NCA model architecture with a two layer update model"""

    def __init__(
        self, n_channels=16, live_threshold=0.1, fire_rate=0.5, hidden_dim=128
    ):
        super().__init__(
            n_channels=n_channels,
            live_threshold=live_threshold,
            fire_rate=fire_rate,
            hidden_dim=hidden_dim,
        )

    def construct_update_model(self):
        return keras.Sequential(
            [
                keras.layers.Conv2D(self.hidden_dim, 1, activation=tf.nn.relu),
                keras.layers.Conv2D(self.hidden_dim, 1, activation=tf.nn.relu),
                keras.layers.Conv2D(
                    self.n_channels,
                    1,
                    activation=None,
                    kernel_initializer="zeros",
                ),
            ]
        )


class BZ_NCA(MorphogenesisNCA):
    def __init__(self, n_channels=3, fire_rate=0.9, hidden_dim=128):
        super().__init__(
            n_channels=n_channels,
            live_threshold=0,
            fire_rate=fire_rate,
            hidden_dim=hidden_dim,
        )

    def get_living_mask(self, x):
        # Return all true of same shape as x
        return tf.ones_like(x, dtype=tf.bool)


class BZ_AverageFilter_NCA(AveragingPerceptionMixin, BZ_NCA):
    def __init__(self, n_channels=3, fire_rate=0.9, hidden_dim=128):
        BZ_NCA.__init__(
            self,
            n_channels=n_channels,
            fire_rate=fire_rate,
            hidden_dim=hidden_dim,
        )


class BZ_AveragingSobel_NCA(AveragingSobelPerceptionMixin, BZ_NCA):
    def __init__(self, n_channels=3, fire_rate=0.9, hidden_dim=128):
        BZ_NCA.__init__(
            self, n_channels=n_channels, fire_rate=fire_rate, hidden_dim=hidden_dim
        )


class GoL_NCA(MorphogenesisNCA):
    def __init__(self, fire_rate=1.0, live_threshold=0.5, hidden_dim=128):
        super().__init__(
            n_channels=1,
            live_threshold=live_threshold,
            fire_rate=fire_rate,
            hidden_dim=hidden_dim,
        )

    def get_living_mask(self, x):
        return (
            tf.nn.max_pool2d(x, ksize=3, strides=[1, 1, 1, 1], padding="SAME")
            >= self.live_threshold
        )

    @tf.function
    def call(self, x):
        x = tf.where(x < self.live_threshold, tf.zeros_like(x), tf.ones_like(x))
        return super().call(x)


class GoL_AveragingSobel_NCA(AveragingSobelPerceptionMixin, GoL_NCA):
    def __init__(self, fire_rate=1.0, live_threshold=0.5, hidden_dim=128):
        GoL_NCA.__init__(
            self,
            fire_rate=fire_rate,
            live_threshold=live_threshold,
            hidden_dim=hidden_dim,
        )


class LearnablePerception_MorphogenesisNCA(LearnablePerceptionMixin, MorphogenesisNCA):
    def __init__(
        self,
        n_channels=16,
        live_threshold=0.1,
        fire_rate=0.5,
        n_filters=3,
        hidden_dim=128,
    ):
        self.n_filters = n_filters
        MorphogenesisNCA.__init__(
            self,
            n_channels=n_channels,
            live_threshold=live_threshold,
            fire_rate=fire_rate,
            hidden_dim=hidden_dim,
        )


class LearnablePerception_BZ_NCA(LearnablePerceptionMixin, BZ_NCA):
    def __init__(self, n_channels=3, fire_rate=0.9, n_filters=3, hidden_dim=128):
        self.n_filters = n_filters
        BZ_NCA.__init__(
            self, n_channels=n_channels, fire_rate=fire_rate, hidden_dim=hidden_dim
        )


class Learnable_GoL_NCA(LearnablePerceptionMixin, GoL_NCA):
    def __init__(
        self,
        fire_rate=1.0,
        live_threshold=0.5,
        n_filters=3,
        hidden_dim=128,
    ):
        self.n_filters = n_filters
        GoL_NCA.__init__(
            self,
            fire_rate=fire_rate,
            live_threshold=live_threshold,
            hidden_dim=hidden_dim,
        )
