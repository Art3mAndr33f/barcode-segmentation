"""
Модуль для инференса моделей.
"""

from .predictor import BarcodePredictor
from .postprocessing import PostProcessor

__all__ = ["BarcodePredictor", "PostProcessor"]
