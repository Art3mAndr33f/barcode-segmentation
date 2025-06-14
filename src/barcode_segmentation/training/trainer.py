#!/usr/bin/env python3
"""
Trainer модуль для обучения модели сегментации штрих-кодов.

Использует PyTorch Lightning для организации процесса обучения и интегрируется с MLflow 
для отслеживания экспериментов. Поддерживает настройку через Hydra.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

from barcode_segmentation.data.dataloader import BarcodeDataModule
from barcode_segmentation.models.lightning_module import BarcodeLightningModule
from barcode_segmentation.training.evaluator import ModelEvaluator
from barcode_segmentation.utils.metrics import compute_metrics

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BarcodeTrainer:
    """
    Класс для обучения модели сегментации штрих-кодов.
    
    Использует PyTorch Lightning для организации процесса обучения и интегрируется 
    с MLflow для отслеживания экспериментов. Поддерживает настройку через Hydra.
    
    Attributes:
        config: Конфигурация для обучения
        model: Модель Lightning
        data_module: Модуль данных Lightning
        trainer: Trainer Lightning
        evaluator: Класс для оценки модели
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Инициализация тренера.
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config = None
        self.model = None
        self.data_module = None
        self.trainer = None
        self.evaluator = None
        self.config_path = config_path
        
    def run(self, config_path: Optional[str] = None) -> None:
        """
        Запуск полного процесса обучения.
        
        Args:
            config_path: Путь к конфигурационному файлу (опционально)
        """
        # Используем config_path из аргумента или из инициализации
        config_path = config_path or self.config_path or "configs/train.yaml"
        
        # Загружаем конфигурацию через Hydra
        with hydra.initialize_config_module(config_module="configs"):
            self.config = hydra.compose(config_name=Path(config_path).stem)
            
        logger.info(f"Загружена конфигурация: {config_path}")
        
        try:
            # Настраиваем окружение
            self._setup_environment()
            
            # Инициализируем компоненты
            self._initialize_data_module()
            self._initialize_model()
            self._initialize_trainer()
            
            # Запускаем обучение
            self._train()
            
            # Оцениваем модель
            self._evaluate()
            
            # Сохраняем результаты
            self._save_results()
            
            logger.info("✅ Обучение завершено успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении: {e}")
            raise
            
    def _setup_environment(self) -> None:
        """Настройка окружения для обучения."""
        # Устанавливаем seed для воспроизводимости
        pl.seed_everything(self.config.seed, workers=True)
        
        # Настраиваем MLflow
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        
        # Создаем директории для сохранения результатов
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_dir = Path(self.config.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Логируем используемую конфигурацию
        logger.info(f"Используемая конфигурация:\n{OmegaConf.to_yaml(self.config)}")
        
    def _initialize_data_module(self) -> None:
        """Инициализация модуля данных."""
        logger.info("Инициализация модуля данных")
        
        # Создаем data module
        self.data_module = BarcodeDataModule(
            data_dir=self.config.data_dir,
            batch_size=self.config.training.batch_size,
            num_workers=self.config.training.num_workers,
            train_transforms=self.config.data.train_transforms,
            val_transforms=self.config.data.val_transforms
        )
        
        # Подготавливаем данные
        self.data_module.prepare_data()
        self.data_module.setup(stage="fit")
        
        logger.info(f"✓ Data module инициализирован")
        logger.info(f"✓ Тренировочных образцов: {len(self.data_module.train_dataset)}")
        logger.info(f"✓ Валидационных образцов: {len(self.data_module.val_dataset)}")
        
    def _initialize_model(self) -> None:
        """Инициализация модели."""
        logger.info("Инициализация модели")
        
        # Создаем модель
        self.model = BarcodeLightningModule(
            model_config=self.config.model,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            lr_scheduler=self.config.training.scheduler
        )
        
        logger.info("✓ Модель инициализирована")
        
    def _initialize_trainer(self) -> None:
        """Инициализация тренера PyTorch Lightning."""
        logger.info("Инициализация тренера")
        
        # Настраиваем логгер MLflow
        mlf_logger = MLFlowLogger(
            experiment_name=self.config.mlflow.experiment_name,
            tracking_uri=self.config.mlflow.tracking_uri
        )
        
        # Логируем параметры модели и тренировки
        mlf_logger.log_hyperparams(self.config)
        
        # Настраиваем callbacks
        callbacks = [
            # Мониторинг learning rate
            LearningRateMonitor(logging_interval="step"),
            
            # Checkpoint
            ModelCheckpoint(
                dirpath=os.path.join(self.config.model_dir, "checkpoints"),
                filename="{epoch}-{val_loss:.4f}",
                monitor=self.config.checkpoint.monitor,
                mode=self.config.checkpoint.mode,
                save_top_k=self.config.checkpoint.save_top_k,
                every_n_epochs=self.config.checkpoint.every_n_epochs,
            ),
            
            # Early stopping
            EarlyStopping(
                monitor=self.config.early_stopping.monitor,
                patience=self.config.early_stopping.patience,
                mode=self.config.early_stopping.mode,
            )
        ]
        
        # Создаем тренер
        self.trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            val_check_interval=self.config.training.val_check_interval,
            accelerator=self.config.training.accelerator,
            devices=self.config.training.devices,
            precision=self.config.training.precision,
            gradient_clip_val=self.config.training.gradient_clip_val,
            logger=mlf_logger,
            callbacks=callbacks,
            log_every_n_steps=self.config.logging.log_every_n_steps,
            default_root_dir=self.config.output_dir,
        )
        
        logger.info("✓ Тренер инициализирован")
        
    def _train(self) -> None:
        """Запуск процесса обучения."""
        logger.info("🚀 Запуск обучения")
        
        # Запускаем обучение
        self.trainer.fit(self.model, self.data_module)
        
        # Сохраняем best checkpoint
        best_model_path = self.trainer.checkpoint_callback.best_model_path
        if best_model_path:
            logger.info(f"✓ Лучшая модель сохранена: {best_model_path}")
        
    def _evaluate(self) -> Dict[str, Any]:
        """
        Оценка модели после обучения.
        
        Returns:
            Словарь с метриками
        """
        logger.info("📊 Оценка модели")
        
        # Инициализируем evaluator
        self.evaluator = ModelEvaluator(
            model=self.model,
            data_module=self.data_module,
            config=self.config
        )
        
        # Запускаем оценку на валидационном наборе
        metrics = self.evaluator.evaluate()
        
        # Логируем метрики в MLflow
        with mlflow.start_run():
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
        
        logger.info(f"Метрики модели: {metrics}")
        return metrics
        
    def _save_results(self) -> None:
        """Сохранение результатов обучения."""
        logger.info("💾 Сохранение результатов обучения")
        
        # Финальный путь модели
        final_model_path = Path(self.config.model_dir) / "final_model.pt"
        
        # Сохраняем состояние модели
        self.model.save_model(str(final_model_path))
        
        # Логируем модель в MLflow
        with mlflow.start_run():
            mlflow.pytorch.log_model(
                self.model.model,
                "model",
                conda_env={
                    "name": "barcode_segmentation",
                    "channels": ["pytorch", "conda-forge"],
                    "dependencies": [
                        "python>=3.8",
                        "pytorch>=2.0.0",
                        "torchvision>=0.15.0",
                        "detectron2",
                    ]
                }
            )
        
        # Сохраняем конфигурацию
        config_path = Path(self.config.output_dir) / "config.yaml"
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.config))
            
        logger.info(f"✓ Модель сохранена: {final_model_path}")
        logger.info(f"✓ Конфигурация сохранена: {config_path}")


def main() -> None:
    """Основная функция для запуска обучения из командной строки."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Обучение модели сегментации штрих-кодов")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Путь к конфигурации")
    
    args = parser.parse_args()
    
    trainer = BarcodeTrainer()
    trainer.run(args.config)


if __name__ == "__main__":
    main()