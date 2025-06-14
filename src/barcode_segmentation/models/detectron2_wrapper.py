"""
Обертка для интеграции Detectron2 с PyTorch Lightning.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from omegaconf import DictConfig


class Detectron2Wrapper(nn.Module):
    """
    Обертка для модели Detectron2, интегрированная с PyTorch Lightning.
    """

    def __init__(self, config: DictConfig):
        """
        Инициализация обертки Detectron2.

        Args:
            config: Конфигурация модели
        """
        super().__init__()
        self.config = config
        self.cfg = self._setup_detectron2_config()
        self.model = build_model(self.cfg)
        self.model.train()

        # Настройка логгера
        setup_logger()

    def _setup_detectron2_config(self):
        """
        Настройка конфигурации Detectron2.

        Returns:
            Конфигурация Detectron2
        """
        cfg = get_cfg()

        # Загружаем базовую конфигурацию
        config_file = self.config.detectron2.config_file
        cfg.merge_from_file(model_zoo.get_config_file(config_file))

        # Настройки датасета
        cfg.DATASETS.TRAIN = ("barcode_train",)
        cfg.DATASETS.TEST = ("barcode_val",)

        # Настройки DataLoader
        cfg.DATALOADER.NUM_WORKERS = self.config.get("num_workers", 2)

        # Веса модели
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)

        # Параметры модели
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.config.detectron2.num_classes
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = self.config.detectron2.roi_heads.batch_size_per_image
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.detectron2.roi_heads.score_thresh_test

        # Параметры обучения (будут переопределены в Lightning)
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 1000

        # Устройство
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        return cfg

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Прямой проход модели.

        Args:
            batched_inputs: Батч входных данных

        Returns:
            Результат модели
        """
        return self.model(batched_inputs)

    def training_step(self, batched_inputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Шаг тренировки.

        Args:
            batched_inputs: Батч входных данных

        Returns:
            Лосс для оптимизации
        """
        self.model.train()
        losses = self.model(batched_inputs)

        # Суммируем все лоссы
        total_loss = sum(losses.values())

        return total_loss, losses

    def validation_step(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Шаг валидации.

        Args:
            batched_inputs: Батч входных данных

        Returns:
            Результат предсказания
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batched_inputs)
        return outputs

    def predict(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Предсказание модели.

        Args:
            batched_inputs: Батч входных данных

        Returns:
            Результат предсказания
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batched_inputs)
        return outputs

    def get_model_config(self):
        """
        Возвращает конфигурацию модели Detectron2.

        Returns:
            Конфигурация Detectron2
        """
        return self.cfg

    def get_detectron2_model(self):
        """
        Возвращает модель Detectron2.

        Returns:
            Модель Detectron2
        """
        return self.model


class Detectron2Trainer(DefaultTrainer):
    """
    Кастомный тренер Detectron2 для интеграции с Lightning.
    """

    def __init__(self, cfg, lightning_module=None):
        """
        Инициализация тренера.

        Args:
            cfg: Конфигурация Detectron2
            lightning_module: Lightning модуль (опционально)
        """
        super().__init__(cfg)
        self.lightning_module = lightning_module

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Создает оценщик для валидации.

        Args:
            cfg: Конфигурация
            dataset_name: Имя датасета
            output_folder: Папка для вывода

        Returns:
            Оценщик COCO
        """
        if output_folder is None:
            output_folder = "./output/"

        return COCOEvaluator(
            dataset_name, 
            cfg, 
            False, 
            output_dir=output_folder
        )

    def run_evaluation(self, dataset_name: str = "barcode_val") -> Dict:
        """
        Запускает оценку модели на датасете.

        Args:
            dataset_name: Имя датасета для оценки

        Returns:
            Результаты оценки
        """
        evaluator = self.build_evaluator(self.cfg, dataset_name)
        val_loader = build_detection_test_loader(self.cfg, dataset_name)

        results = inference_on_dataset(self.model, val_loader, evaluator)

        return results
