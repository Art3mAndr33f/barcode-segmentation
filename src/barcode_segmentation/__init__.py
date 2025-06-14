"""
Пакет для MLOps проекта сегментации штрих-кодов.

Этот пакет предоставляет полный pipeline машинного обучения для задачи сегментации
штрих-кодов, включая предобработку данных, обучение моделей, инференс и деплоймент.

Основные модули:
- data: Работа с данными и их предобработка
- models: Модели машинного обучения и их обертки
- training: Компоненты для обучения моделей
- inference: Компоненты для инференса
- deployment: Инструменты для деплоймента в продакшен
- utils: Вспомогательные утилиты

Пример использования:
    from barcode_segmentation.data import BarcodeDataModule
    from barcode_segmentation.models import BarcodeLightningModule
    from barcode_segmentation.training import BarcodeTrainer
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
__description__ = "MLOps проект для автоматической сегментации штрих-кодов"

# Основные экспорты
from barcode_segmentation.data import BarcodeDataModule, BarcodeDataset, DataPreprocessor
from barcode_segmentation.models import BarcodeLightningModule, Detectron2Wrapper
from barcode_segmentation.training import BarcodeTrainer, ModelEvaluator
from barcode_segmentation.inference import BarcodePredictor, PostProcessor
from barcode_segmentation.deployment import ONNXConverter, TensorRTConverter

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    # Data
    "BarcodeDataModule",
    "BarcodeDataset", 
    "DataPreprocessor",
    # Models
    "BarcodeLightningModule",
    "Detectron2Wrapper",
    # Training
    "BarcodeTrainer",
    "ModelEvaluator", 
    # Inference
    "BarcodePredictor",
    "PostProcessor",
    # Deployment
    "ONNXConverter",
    "TensorRTConverter",
]