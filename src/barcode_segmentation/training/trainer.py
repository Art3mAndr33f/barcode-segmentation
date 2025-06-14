"""
Модуль для тренировки модели с использованием PyTorch Lightning.
"""

import os
from pathlib import Path
from typing import Optional

import hydra
import lightning as L
import mlflow
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor
)
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig

from ..data.dataloader import BarcodeDataModule
from ..models.lightning_module import BarcodeLightningModule


class BarcodeTrainer:
    """Класс для тренировки модели сегментации штрих-кодов."""

    def __init__(self):
        """Инициализация тренера."""
        pass

    def setup_mlflow(self, config: DictConfig) -> MLFlowLogger:
        """
        Настройка MLflow логгера.

        Args:
            config: Конфигурация

        Returns:
            MLflow логгер
        """
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)

        # Создаем или получаем эксперимент
        try:
            experiment_id = mlflow.create_experiment(config.mlflow.experiment_name)
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
            experiment_id = experiment.experiment_id

        # Создаем логгер
        logger = MLFlowLogger(
            experiment_name=config.mlflow.experiment_name,
            tracking_uri=config.mlflow.tracking_uri,
            log_model=True
        )

        return logger

    def setup_callbacks(self, config: DictConfig) -> list:
        """
        Настройка колбэков для тренировки.

        Args:
            config: Конфигурация

        Returns:
            Список колбэков
        """
        callbacks = []

        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(config.output_dir) / "checkpoints",
            filename="barcode-{epoch:02d}-{val_loss:.2f}",
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_top_k=config.checkpoint.save_top_k,
            every_n_epochs=config.checkpoint.every_n_epochs,
            save_last=True
        )
        callbacks.append(checkpoint_callback)

        # Early stopping
        early_stopping = EarlyStopping(
            monitor=config.early_stopping.monitor,
            patience=config.early_stopping.patience,
            mode=config.early_stopping.mode,
            verbose=True
        )
        callbacks.append(early_stopping)

        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

        return callbacks

    def train_model(self, config: DictConfig):
        """
        Основная функция тренировки модели.

        Args:
            config: Конфигурация обучения
        """
        # Устанавливаем seed для воспроизводимости
        L.seed_everything(config.seed)

        # Настраиваем данные
        data_module = BarcodeDataModule.from_config(config)
        data_module.prepare_data()
        data_module.setup("fit")

        # Создаем модель
        model = BarcodeLightningModule(config)

        # Настраиваем логгер
        logger = self.setup_mlflow(config)

        # Настраиваем колбэки
        callbacks = self.setup_callbacks(config)

        # Создаем тренер
        trainer = L.Trainer(
            max_epochs=config.training.max_epochs,
            accelerator=config.training.accelerator,
            devices=config.training.devices,
            precision=config.training.precision,
            logger=logger,
            callbacks=callbacks,
            val_check_interval=config.training.val_check_interval,
            gradient_clip_val=config.training.gradient_clip_val,
            default_root_dir=config.output_dir,
            enable_checkpointing=True,
            log_every_n_steps=config.logging.log_every_n_steps
        )

        # Логируем конфигурацию и git commit
        self._log_experiment_info(config, logger)

        print("Начинаем тренировку...")
        trainer.fit(model, datamodule=data_module)

        print("Тренировка завершена!")

        # Сохраняем финальную модель
        self._save_final_model(trainer, model, config)

        return trainer, model

    def _log_experiment_info(self, config: DictConfig, logger: MLFlowLogger):
        """
        Логирует информацию об эксперименте.

        Args:
            config: Конфигурация
            logger: MLflow логгер
        """
        # Логируем гиперпараметры
        hyperparams = {
            "model_architecture": config.model.architecture,
            "backbone": config.model.backbone,
            "num_classes": config.model.detectron2.num_classes,
            "learning_rate": config.optimizer.lr,
            "batch_size": config.data_loading.batch_size,
            "max_epochs": config.training.max_epochs,
            "train_split": config.dataset.train_split,
            "val_split": config.dataset.val_split
        }

        for key, value in hyperparams.items():
            logger.experiment.log_param(logger.run_id, key, value)

        # Логируем git commit если возможно
        try:
            import subprocess
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            ).decode("utf-8").strip()
            logger.experiment.log_param(logger.run_id, "git_commit", git_commit)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Не удалось получить git commit")

    def _save_final_model(self, trainer: L.Trainer, 
                         model: BarcodeLightningModule, 
                         config: DictConfig):
        """
        Сохраняет финальную модель.

        Args:
            trainer: Тренер Lightning
            model: Модель
            config: Конфигурация
        """
        model_dir = Path(config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем лучшую модель
        best_model_path = model_dir / "best_model.ckpt"
        if trainer.checkpoint_callback.best_model_path:
            import shutil
            shutil.copy2(trainer.checkpoint_callback.best_model_path, best_model_path)
            print(f"Лучшая модель сохранена: {best_model_path}")

        # Сохраняем последнюю модель
        last_model_path = model_dir / "last_model.ckpt"
        trainer.save_checkpoint(last_model_path)
        print(f"Последняя модель сохранена: {last_model_path}")

    @hydra.main(version_base=None, config_path="../../configs", config_name="train")
    def run(self, config: DictConfig):
        """
        Запуск тренировки с конфигурацией Hydra.

        Args:
            config: Конфигурация Hydra
        """
        print("Запуск тренировки модели...")

        # Создаем необходимые директории
        for dir_path in [config.output_dir, config.model_dir, config.plots_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Проверяем наличие данных
        data_dir = Path(config.data_dir) / "processed"
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Обработанные данные не найдены в {data_dir}. "
                "Запустите сначала предобработку данных."
            )

        # Запускаем тренировку
        trainer, model = self.train_model(config)

        print("Тренировка завершена успешно!")
        return trainer, model


def main():
    """Точка входа для тренировки."""
    trainer = BarcodeTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
