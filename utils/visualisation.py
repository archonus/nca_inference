"""Visualisation utilities for NCA inference
This module contains functions for visualising images, videos, and other data."""

import io
import os
from typing import List, Optional

import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import Tensor
import IPython.display as ipython_display
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from numpy.typing import ArrayLike, NDArray
from PIL.Image import Image as PILImage

os.environ["FFMPEG_BINARY"] = "ffmpeg"

from nca_inference.utils import raise_if


def convert_to_image(data: NDArray) -> PILImage:
    """
    Convert a numpy array to a PIL image

    Args:
        data (NDArray): Numpy array containing image data.

    Returns:
        Image: PIL Image object.
    """
    if data.dtype in [np.float32, np.float64]:
        data = np.uint8(np.clip(data, 0, 1) * 255)
    if (
        len(data.shape) == 3 and data.shape[-1] == 1
    ):  # Grey-scale but has channel dimension
        data = np.repeat(data, 3, -1)
    return PIL.Image.fromarray(data)


def save_image(f, a, fmt=None):
    a = np.asarray(a)
    if isinstance(f, str):
        fmt = f.rsplit(".", 1)[-1].lower()
        if fmt == "jpg":
            fmt = "jpeg"
        f = open(f, "wb")
    convert_to_image(a).save(f, fmt, quality=95)


def encode_image(a: ArrayLike, fmt="jpeg") -> bytes:
    """Encode image to bytes"""
    a = np.asarray(a)
    if len(a.shape) == 3 and a.shape[-1] == 4:
        fmt = "png"
    f = io.BytesIO()
    save_image(f, a, fmt)
    return f.getvalue()


def display_image(a: ArrayLike, fmt="jpeg"):
    ipython_display.display(ipython_display.Image(data=encode_image(a, fmt)))


def horizontal_stack(vis: ArrayLike, spacing=5, pad_bottom=False) -> ArrayLike:
    """
    Stack images horizontally

    Args:
        vis (ArrayLike): Batch of images to stack, of shape (b, h, w, c)
        spacing (int, optional): Spacing between images. Defaults to 5.
        pad_bottom (bool, optional): Whether to include padding on the bottom of the image. Defaults to False.

    Returns:
        ArrayLike: Stacked images. Will have shape (h, b*w, c) if no spacing
    """
    raise_if(len(vis.shape) != 4, "Input must be a batch of images (b, h, w, c)")

    if spacing > 0:
        vis = np.pad(
            vis,
            ((0, 0), (0, spacing if pad_bottom else 0), (0, spacing), (0, 0)),
            "constant",
            constant_values=255,
        )

    # Stack horizontally in batch dimension to get (h, b*w, c)
    vis = np.hstack(vis)
    if spacing > 0:
        vis = vis[:, :-spacing, :]
    return vis


def zoom(img: ArrayLike, scale=4):
    """Zoom in on an image by a factor of scale"""
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


def to_rgba(x: ArrayLike) -> ArrayLike:
    """Extract RGBA channels from image data"""
    return x[..., :4]


def to_alpha(x: ArrayLike) -> ArrayLike:
    """
    Extract alpha channel from image data

    Args:
        x (ArrayLike): Image data of shape (..., >= 4)

    Returns:
        ArrayLike: Alpha channel of shape (..., 1)
    """
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)


def to_rgb(x: ArrayLike) -> ArrayLike:
    """
    Convert image data to RGB format
    This function converts an image with an alpha channel to RGB format by blending the RGB channels with the alpha channel.

    Args:
        x (ArrayLike): Image data of shape (..., c) where first three channels are RGB and the fourth channel is alpha.

    Returns:
        ArrayLike: _description_
    """
    if x.shape[-1] <= 3:
        return x
    rgb, a = x[..., :3], to_alpha(x)
    return 1.0 - a + rgb


def show_single_chanel(frames: List[Tensor], out_filename: str, channel: int = 4):
    """
    Visualise a single channel of the image data

    Args:
        frames (List[Tensor]): List of frames to visualise.
        out_filename (str): Output file name.
        channel (int, optional): Channel to visualise. Defaults to 4.
    """
    with VideoWriter(out_filename) as vid:
        for frame in frames:
            vis = frame[..., channel : channel + 1]
            vid.add(zoom(vis, 2))
    return out_filename


def visualise_channels(frames: List[Tensor], out_filename: str, show_alpha_sep=True):
    """
    Visualise all channels of the image data as a video

    Args:
        frames (List[Tensor]): List of frames to visualise.
        out_filename (str): Output file name.
        show_alpha_sep (bool, optional): Whether to display the alpha channel separately. Defaults to True.
    """
    n_channels = frames[0].shape[-1]
    raise_if(n_channels < 4, "Frames must contain at least RGBA channels")
    with VideoWriter(out_filename) as vid:
        for frame in frames:
            rgb = frame[..., :3]
            alpha = to_alpha(frame)
            vis = [1.0 - alpha + rgb]
            if show_alpha_sep:
                start = 4
                vis.append(np.repeat(alpha, 3, -1))
            else:
                start = 3
            for k in range(start, n_channels, 3):
                # vis.append(1.0 - alpha + frame[..., k:k+3])
                vis.append(frame[..., k : k + 3])
            vid.add(zoom(np.hstack(vis), 2))


class VideoWriter:
    """Class for writing frames to a video file"""

    def __init__(self, filename, fps=30.0, **kw):
        self.writer: Optional[FFMPEG_VideoWriter] = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img: ArrayLike):
        """Add a single frame (colour or grayscale) to video"""
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:  # Grayscale
            img = np.repeat(img[..., None], 3, -1)
        elif (
            len(img.shape) == 3 and img.shape[-1] == 1
        ):  # Grayscale with channel dimension
            img = np.repeat(img, 3, -1)
        self.writer.write_frame(img)

    def add_frames(self, frames: ArrayLike, scale=2):
        """
        Add a batch of frames to the video.

        Args:
            frames (ArrayLike): Batch of frames to add, of shape (b, h, w, c)
            scale (int, optional): Scale to zoom each frame by. Defaults to 2.
        """
        for frame in frames:
            vis = to_rgb(frame)
            self.add(zoom(vis, scale))

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()

    @staticmethod
    def gen_video(frames: ArrayLike, out_name: str, scale=2, fps=30.0):
        """
        Generate a video from a batch of frames.

        Args:
            frames (ArrayLike): Batch of frames to add, of shape (b, h, w, c).
            out_name (str): Output file name.
            scale (int, optional): Scale to zoom each frame by. Defaults to 2.
            fps (float, optional): Frames per second. Defaults to 30.0.
        """
        with VideoWriter(out_name, fps=fps) as vid:
            vid.add_frames(frames, scale=scale)
