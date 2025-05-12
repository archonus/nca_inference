import io
from typing import Optional

import numpy as np
import PIL
import requests
import tensorflow as tf
from numpy.typing import NDArray
from tensorflow import Tensor

from .visualisation import display_image, to_rgb, zoom

FOX_EMOJI = "🦊"
SMILEY_FACE_EMOJI = "🙃"
CAT_EMOJI = "🐱"


class ImageLoader:
    """Utility class to load images from URLs and generate random seeds for NCA."""

    def __init__(self, n_channels=16, target_padding=16, target_size=40):
        self.n_channels = n_channels
        self.target_padding = target_padding
        self.target_size = target_size

    def load_image(self, url: str, target_size=None):
        if target_size is None:
            target_size = self.target_size
        r = requests.get(url, timeout=10)
        img = PIL.Image.open(io.BytesIO(r.content))
        img.thumbnail((target_size, target_size), PIL.Image.Resampling.LANCZOS)
        img = np.float32(img) / 255.0
        # premultiply RGB by Alpha
        img[..., :3] *= img[..., 3:]
        return img

    def load_emoji(self, emoji):
        code = hex(ord(emoji))[2:].lower()
        url = f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{code}.png?raw=true"
        return self.load_image(url)

    def pad_target(self, target_img):
        p = self.target_padding
        return tf.pad(
            target_img, [(p, p), (p, p), (0, 0)]
        )  # Do not pad in channel dimension

    def generate_seed(self, h: int, w: int) -> NDArray:
        seed = np.zeros([h, w, self.n_channels], np.float32)
        seed[h // 2, w // 2, 3:] = 1.0
        return seed

    @staticmethod
    @tf.function
    def make_circle_masks(h: int, w: int, n=1) -> Tensor:
        """Return mask of shape (n,h,w)"""
        x = tf.linspace(-1.0, 1.0, w)[np.newaxis, np.newaxis, :]
        y = tf.linspace(-1.0, 1.0, h)[np.newaxis, :, np.newaxis]
        center = tf.random.uniform([2, n, 1, 1], -0.5, 0.5)
        r = tf.random.uniform([n, 1, 1], 0.1, 0.4)
        x, y = (x - center[0]) / r, (y - center[1]) / r
        mask = tf.cast(x * x + y * y < 1.0, tf.float32)
        return mask


def generate_padded_target_and_seed(
    image_loader: Optional[ImageLoader] = None, target_emoji: str = FOX_EMOJI
) -> tuple[NDArray, NDArray]:
    """
    Generate a padded target image and a random seed for NCA.

    Args:
        image_loader (ImageLoader, optional): ImageLoader instance. Defaults to None.
        target_emoji (str, optional): Target emoji to load. Defaults to FOX_EMOJI.

    Returns:
        tuple[NDArray, NDArray]: A tuple containing the padded target image and the random seed.
    """
    if image_loader is None:
        image_loader = ImageLoader()
    target_img = image_loader.load_emoji(target_emoji)
    display_image(zoom(to_rgb(target_img), 2), fmt="png")
    padded_target = image_loader.pad_target(target_img)
    h, w = padded_target.shape[:2]
    seed = image_loader.generate_seed(h, w)
    return padded_target, seed
