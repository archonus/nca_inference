from typing import List

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve


def bz_update(
    frame: NDArray, alpha: float, beta: float, gamma: float, mode: str = "wrap"
) -> NDArray:
    """Update the frame using the given alpha, beta, gamma values using the model by Turner (2009)."""
    if frame.shape[-1] != 3:
        raise ValueError("Frame must have exactly 3 channels")
    averaging = np.ones((3, 3, 1), dtype=np.float32) / 9
    averages = convolve(frame, averaging, mode=mode)
    A = averages[..., 0]
    B = averages[..., 1]
    C = averages[..., 2]

    next_frame = np.zeros_like(frame)

    next_frame[..., 0] = A + A * (alpha * B - gamma * C)
    next_frame[..., 1] = B + B * (beta * C - alpha * A)
    next_frame[..., 2] = C + C * (gamma * A - beta * B)

    return next_frame.clip(0, 1)


def generate_bz_frames(
    N: int, h: int, w: int, alpha=1.2, beta=1.0, gamma=1.0
) -> List[NDArray]:
    """
    Generates a sequence of frames simulating the Belousov-Zhabotinsky (BZ) reaction.

    Args:
        N (int): The number of frames to generate.
        h (int): The height of each frame.
        w (int): The width of each frame.
        alpha (float, optional): The rate parameter for the first channel. Defaults to 1.2.
        beta (float, optional): The rate parameter for the second channel. Defaults to 1.0.
        gamma (float, optional): The rate parameter for the third channel. Defaults to 1.0.

    Returns:
        List[NDArray]: A list of frames, where each frame is a 3D NumPy array of shape (h, w, 3)
                          representing the state of the reaction at a given time step.
    """
    np.random.seed(0)
    arr = np.random.random_sample(size=(h, w, 3)).astype(np.float32)

    frames = [arr]

    for _ in range(N):
        frames.append(bz_update(frames[-1], alpha=alpha, beta=beta, gamma=gamma))

    return frames
