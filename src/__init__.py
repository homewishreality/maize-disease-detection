"""Source package for the maize disease detection project.

Importing this package never imports TensorFlow, so lightweight utilities
(dataset inspection, prediction metadata) stay fast; heavy modules are pulled
in explicitly (``from src import models``).
"""

__all__ = ["ProjectConfig", "DISPLAY_NAMES", "available_memory_mb"]

from .config import DISPLAY_NAMES, ProjectConfig  # noqa: E402,F401
