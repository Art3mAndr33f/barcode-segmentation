#!/usr/bin/env python3
"""
PyTorch Lightning модуль для модели сегментации штрих-кодов.

Содержит логику обучения, валидации и тестирования модели с использованием Detectron2.
Интегрируется с MLflow для логирования метрик и артефактов.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from barcode_segmentation.models.detectron2_wrapper import Detectron2Wrapper
from barcode_segmentation.utils.metrics import BarcodeMetrics

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BarcodeLightningModule(pl.LightningModule):
    """
    PyTorch Lightning модуль для модели сегментации штрих-кодов.
    
    Содержит полную логику обучения, валидации и тестирования модели.
    Использует Detectron2 для сегментации и поддерживает различные метрики.
    
    Attributes:
        model: Обертка над Detectron2 моделью
        learning_rate: Скорость обучения
        weight_decay: Параметр regularization
        metrics: Класс для вычисления метрик
    """
    
    def __init__(
        self,
        model_config: DictConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        lr_scheduler: Optional[str] = "cosine",
        optimizer: str = "adamw",
        **kwargs
    ):
        """
        Инициализация Lightning модуля.
        
        Args:
            model_config: Конфигурация модели
            learning_rate: Скорость обучения
            weight_decay: Коэффициент weight decay
            lr_scheduler: Тип scheduler ("cosine", "step", None)
            optimizer: Тип оптимизатора ("adamw", "adam", "sgd")
        """
        super().__init__()
        
        # Сохраняем гиперпараметры
        self.save_hyperparameters()
        
        # Инициализируем модель
        self.model = Detectron2Wrapper(model_config)
        
        # Параметры обучения
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.optimizer_name = optimizer
        
        # Метрики
        self.metrics = BarcodeMetrics()
        
        # Для хранения результатов валидации
        self.validation_step_outputs = []
        
        logger.info(f"Инициализирован BarcodeLightningModule с lr={learning_rate}")
        
    def forward(self, images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass модели.
        
        Args:
            images: Список изображений
            
        Returns:
            Список предсказаний
        """
        return self.model(images)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """
        Один шаг обучения.
        
        Args:
            batch: Батч данных
            batch_idx: Индекс батча
            
        Returns:
            Значение loss
        """
        images = batch["images"]
        targets = batch["targets"]
        
        # Forward pass
        loss_dict = self.model.forward_train(images, targets)
        
        # Суммарный loss
        total_loss = sum(loss_dict.values())
        
        # Логируем метрики
        for key, value in loss_dict.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Один шаг валидации.
        
        Args:
            batch: Батч данных
            batch_idx: Индекс батча
            
        Returns:
            Словарь с результатами валидации
        """
        images = batch["images"]
        targets = batch["targets"]
        
        # Forward pass для валидации (inference mode)
        predictions = self.model.forward_inference(images)
        
        # Вычисляем метрики
        metrics = self.metrics.compute_batch_metrics(predictions, targets)
        
        # Сохраняем результаты
        output = {
            "predictions": predictions,
            "targets": targets,
            "metrics": metrics
        }
        
        self.validation_step_outputs.append(output)
        
        # Логируем метрики
        for key, value in metrics.items():
            self.log(f"val_{key}", value, on_step=False, on_epoch=True, prog_bar=True)
        
        return output
    
    def on_validation_epoch_end(self) -> None:
        """Обработка результатов валидации в конце эпохи."""
        if not self.validation_step_outputs:
            return
            
        # Собираем все предсказания и targets
        all_predictions = []
        all_targets = []
        
        for output in self.validation_step_outputs:
            all_predictions.extend(output["predictions"])
            all_targets.extend(output["targets"])
        
        # Вычисляем агрегированные метрики
        epoch_metrics = self.metrics.compute_epoch_metrics(all_predictions, all_targets)
        
        # Логируем метрики эпохи
        for key, value in epoch_metrics.items():
            self.log(f"val_epoch_{key}", value, on_epoch=True, prog_bar=True)
        
        # Очищаем выходы
        self.validation_step_outputs.clear()
        
        logger.info(f"Валидационные метрики эпохи: {epoch_metrics}")
    
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Один шаг тестирования.
        
        Args:
            batch: Батч данных
            batch_idx: Индекс батча
            
        Returns:
            Словарь с результатами тестирования
        """
        images = batch["images"]
        targets = batch["targets"]
        
        # Forward pass
        predictions = self.model.forward_inference(images)
        
        # Вычисляем метрики
        metrics = self.metrics.compute_batch_metrics(predictions, targets)
        
        # Логируем метрики
        for key, value in metrics.items():
            self.log(f"test_{key}", value, on_step=False, on_epoch=True)
        
        return {
            "predictions": predictions,
            "targets": targets,
            "metrics": metrics
        }
    
    def predict_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Шаг предсказания.
        
        Args:
            batch: Батч данных
            batch_idx: Индекс батча
            dataloader_idx: Индекс dataloader
            
        Returns:
            Список предсказаний
        """
        images = batch["images"]
        return self.model.forward_inference(images)
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Настройка оптимизатора и scheduler.
        
        Returns:
            Словарь с конфигурацией оптимизатора
        """
        # Выбираем оптимизатор
        if self.optimizer_name.lower() == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Неподдерживаемый оптимизатор: {self.optimizer_name}")
        
        # Настраиваем scheduler
        if self.lr_scheduler is None:
            return optimizer
        
        if self.lr_scheduler.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.learning_rate * 0.01
            )
        elif self.lr_scheduler.lower() == "step":
            scheduler = StepLR(
                optimizer,
                step_size=self.trainer.max_epochs // 3,
                gamma=0.1
            )
        else:
            raise ValueError(f"Неподдерживаемый scheduler: {self.lr_scheduler}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate"
            }
        }
    
    def on_train_epoch_start(self) -> None:
        """Обработка начала эпохи обучения."""
        logger.info(f"Начало эпохи {self.current_epoch + 1}/{self.trainer.max_epochs}")
        
    def on_train_epoch_end(self) -> None:
        """Обработка конца эпохи обучения."""
        # Логируем learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", current_lr, on_epoch=True)
        
        logger.info(f"Конец эпохи {self.current_epoch + 1}, lr: {current_lr:.6f}")
    
    def save_model(self, path: str) -> None:
        """
        Сохранение модели.
        
        Args:
            path: Путь для сохранения
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'hyperparameters': self.hparams,
            'epoch': self.current_epoch,
        }, path)
        
        logger.info(f"Модель сохранена: {path}")
    
    def load_model(self, path: str) -> None:
        """
        Загрузка модели.
        
        Args:
            path: Путь к модели
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Модель загружена: {path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Получение сводки модели.
        
        Returns:
            Словарь с информацией о модели
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024),  # 4 bytes per parameter
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer_name,
            "lr_scheduler": self.lr_scheduler
        }


def main() -> None:
    """Основная функция для тестирования модуля."""
    # Создаем тестовую конфигурацию
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        "model_name": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "num_classes": 1,
        "score_thresh_test": 0.5,
        "nms_thresh": 0.5
    })
    
    # Создаем модель
    model = BarcodeLightningModule(config)
    
    # Выводим сводку
    summary = model.get_model_summary()
    print("Сводка модели:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()