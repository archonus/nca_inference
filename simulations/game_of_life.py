import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve


patterns = {
    "glider": [(1, 2), (2, 3), (3, 1), (3, 2), (3, 3)],
    "pulsar": [
        (2, 4), (2, 5), (2, 6), (2, 10), (2, 11), (2, 12),
        (4, 2), (4, 7), (4, 9), (4, 14),
        (5, 2), (5, 7), (5, 9), (5, 14),
        (6, 2), (6, 7), (6, 9), (6, 14),
        (7, 4), (7, 5), (7, 6), (7, 10), (7, 11), (7, 12),
        (9, 4), (9, 5), (9, 6), (9, 10), (9, 11), (9, 12),
        (10, 2), (10, 7), (10, 9), (10, 14),
        (11, 2), (11, 7), (11, 9), (11, 14),
        (12, 2), (12, 7), (12, 9), (12, 14),
        (14, 4), (14, 5), (14, 6), (14, 10), (14, 11), (14, 12)
    ],
    "glider_gun": [
        (5, 1),
        (5, 2),
        (6, 1),
        (6, 2),
        (5, 11),
        (6, 11),
        (7, 11),
        (4, 12),
        (8, 12),
        (3, 13),
        (9, 13),
        (3, 14),
        (9, 14),
        (6, 15),
        (4, 16),
        (8, 16),
        (5, 17),
        (6, 17),
        (7, 17),
        (6, 18),
        (3, 21), (4, 21), (5, 21),
        (3, 22), (4, 22), (5, 22),
        (2, 23), (6, 23),
        (1, 25), (2, 25), (6, 25), (7, 25),
        (3, 35), (4, 35),
        (3, 36), (4, 36)
    ],
    "puffer": [
        (5, 5), (5, 6), (6, 5), (6, 6),
        (4, 8), (4, 9), (5, 9), (6, 9), (7, 9), (8, 8),
        (3, 10), (4, 11), (5, 12), (6, 12), (7, 12), (8, 11), (9, 10)
    ]
}

counting_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])


def game_of_life_step(grid: NDArray) -> NDArray:
    """Compute a single step of Conway's Game of Life.

    Args:
        grid (NDArray): Current grid state.

    Returns:
        NDArray: Updated grid state.
    """
    neighbour_count = convolve(grid, counting_kernel, mode="constant", cval=0)
    return np.where(
        (neighbour_count == 3) | ((grid == 1) & (neighbour_count == 2)), 1.0, 0.0
    )


def game_of_life(grid, steps=10):
    """Run Conway's Game of Life for a number of steps.

    Args:
        grid (NDArray): Initial grid state.
        steps (int): Number of steps to run.

    Returns:
        list: List of grid states at each step.
    """
    states = [grid.copy()]
    for _ in range(steps):
        states.append(game_of_life_step(states[-1]))
    return states


def get_pattern(pattern_name: str, h=64, w=64, x_offset=0, y_offset=0) -> NDArray:
    """Get a pattern for Conway's Game of Life.

    Args:
        pattern_name (str): Name of the pattern.
        h (int, optional): Height of the grid. Defaults to 64.
        w (int, optional): Width of the grid. Defaults to 64.
        x_offset (int, optional): X offset for the pattern. Defaults to 0.
        y_offset (int, optional): Y offset for the pattern. Defaults to 0.

    Returns:
        NDArray: Pattern grid of shape (h,w).
    """
    if pattern_name not in patterns:
        raise ValueError(
            f"Pattern '{pattern_name}' not found. Available patterns: {list(patterns.keys())}"
        )

    pattern = np.zeros((h, w), dtype=np.float32)
    for x, y in patterns[pattern_name]:
        pattern[x + x_offset, y + y_offset] = 1.0
    return pattern
