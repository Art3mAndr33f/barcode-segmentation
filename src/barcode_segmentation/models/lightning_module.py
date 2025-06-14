"""
PyTorch Lightning модуль для сегментации штрих-кодов.
"""

from typing import Any, Dict, List, Optional

import lightning as L
import mlflow
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import Accuracy

from .detectron2_wrapper import Detectron2Wrapper


class BarcodeLightningModule(L.LightningModule):
    """
    Lightning модуль для обучения модели сегментации штрих-кодов.
    """

    def __init__(self, config: DictConfig):
        """
        Инициализация Lightning модуля.

        Args:
            config: Конфигурация модели и обучения
        """
        super().__init__()
        self.save_hyperparameters()

        self.config = config
        self.model_wrapper = Detectron2Wrapper(config.model)

        # Метрики
        self.train_losses = []
        self.val_losses = []

        # Настройка автоматического логирования MLflow
        mlflow.pytorch.autolog()

    def forward(self, x):
        """
        Прямой проход модели.

        Args:
            x: Входные данные

        Returns:
            Выход модели
        """
        return self.model_wrapper(x)

    def training_step(self, batch, batch_idx):
        """
        Шаг тренировки.

        Args:
            batch: Батч данных
            batch_idx: Индекс батча

        Returns:
            Лосс
        """
        # Detectron2 ожидает список словарей
        if isinstance(batch, dict):
            batch = [batch]

        total_loss, losses = self.model_wrapper.training_step(batch)

        # Логируем индивидуальные лоссы
        for loss_name, loss_value in losses.items():
            self.log(f"train_{loss_name}", loss_value, 
                    on_step=True, on_epoch=True, prog_bar=True)

        # Логируем общий лосс
        self.log("train_loss", total_loss, 
                on_step=True, on_epoch=True, prog_bar=True)

        self.train_losses.append(total_loss.detach())

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Шаг валидации.

        Args:
            batch: Батч данных
            batch_idx: Индекс батча

        Returns:
            Словарь с метриками
        """
        if isinstance(batch, dict):
            batch = [batch]

        # Для валидации используем модель в режиме eval
        outputs = self.model_wrapper.validation_step(batch)

        # Вычисляем метрики (упрощенно для демонстрации)
        # В реальности здесь нужна более сложная логика оценки
        val_loss = self._compute_validation_loss(batch, outputs)

        self.log("val_loss", val_loss, 
                on_step=False, on_epoch=True, prog_bar=True)

        self.val_losses.append(val_loss.detach())

        return {"val_loss": val_loss}

    def _compute_validation_loss(self, batch, outputs):
        """
        Вычисляет лосс валидации (упрощенная версия).

        Args:
            batch: Батч входных данных
            outputs: Выходы модели

        Returns:
            Лосс валидации
        """
        # Упрощенная версия - в реальности нужна более сложная логика
        # Для демонстрации возвращаем случайное значение
        return torch.tensor(0.5, device=self.device)

    def test_step(self, batch, batch_idx):
        """
        Шаг тестирования.

        Args:
            batch: Батч данных
            batch_idx: Индекс батча

        Returns:
            Словарь с метриками
        """
        if isinstance(batch, dict):
            batch = [batch]

        outputs = self.model_wrapper.predict(batch)

        # Здесь можно добавить вычисление метрик качества
        # например, mAP, IoU и т.д.

        return {"test_outputs": outputs}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Шаг предсказания.

        Args:
            batch: Батч данных
            batch_idx: Индекс батча
            dataloader_idx: Индекс dataloader

        Returns:
            Предсказания модели
        """
        if isinstance(batch, dict):
            batch = [batch]

        return self.model_wrapper.predict(batch)

    def configure_optimizers(self):
        """
        Настройка оптимизатора и планировщика.

        Returns:
            Конфигурация оптимизатора
        """
        # Получаем параметры модели Detectron2
        model = self.model_wrapper.get_detectron2_model()

        # Создаем оптимизатор
        optimizer_config = self.config.optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config.lr,
            weight_decay=optimizer_config.weight_decay,
            betas=optimizer_config.betas,
            eps=optimizer_config.eps
        )

        # Создаем планировщик если указан
        if hasattr(self.config, 'scheduler'):
            scheduler_config = self.config.scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.step_size,
                gamma=scheduler_config.gamma
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss"
                }
            }

        return optimizer

    def on_train_epoch_end(self):
        """Действия в конце эпохи тренировки."""
        if self.train_losses:
            avg_train_loss = torch.stack(self.train_losses).mean()
            self.log("avg_train_loss", avg_train_loss)
            self.train_losses.clear()

    def on_validation_epoch_end(self):
        """Действия в конце эпохи валидации."""
        if self.val_losses:
            avg_val_loss = torch.stack(self.val_losses).mean()
            self.log("avg_val_loss", avg_val_loss)
            self.val_losses.clear()

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Сохранение дополнительной информации в checkpoint.

        Args:
            checkpoint: Словарь checkpoint
        """
        # Сохраняем конфигурацию Detectron2
        checkpoint["detectron2_config"] = self.model_wrapper.get_model_config()

        # Логируем в MLflow
        mlflow.log_param("model_architecture", self.config.model.architecture)
        mlflow.log_param("backbone", self.config.model.backbone)
        mlflow.log_param("num_classes", self.config.model.detectron2.num_classes)

    def get_model_for_export(self):
        """
        Возвращает модель для экспорта в ONNX/TensorRT.

        Returns:
            Модель для экспорта
        """
        return self.model_wrapper.get_detectron2_model()
