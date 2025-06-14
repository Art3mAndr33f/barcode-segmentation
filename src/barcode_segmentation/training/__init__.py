"""
Модуль для тренировки моделей.
"""

from .trainer import BarcodeTrainer
from .evaluator import ModelEvaluator

__all__ = ["BarcodeTrainer", "ModelEvaluator"]
