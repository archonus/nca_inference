"""
Implementations of various types of Neural Cellular Automata (NCA) models.
"""

from .models import (
    BZ_NCA,
    BZ_AverageFilter_NCA,
    BZ_AveragingSobel_NCA,
    GoL_AveragingSobel_NCA,
    GoL_NCA,
    Learnable_GoL_NCA,
    LearnablePerception_BZ_NCA,
    LearnablePerception_MorphogenesisNCA,
    MorphogenesisNCA,
    TwoLayerNCA,
)
from .nca_base import NCA

__all__ = [
    "BZ_NCA",
    "BZ_AverageFilter_NCA",
    "BZ_AveragingSobel_NCA",
    "GoL_AveragingSobel_NCA",
    "GoL_NCA",
    "Learnable_GoL_NCA",
    "LearnablePerception_BZ_NCA",
    "LearnablePerception_MorphogenesisNCA",
    "MorphogenesisNCA",
    "NCA",
    "TwoLayerNCA",
]
