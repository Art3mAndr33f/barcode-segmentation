"""
Тесты для модуля тренировки.
"""

import pytest
import tempfile
from pathlib import Path
from omegaconf import OmegaConf

from barcode_segmentation.training.trainer import BarcodeTrainer


class TestBarcodeTrainer:
    """Тесты для тренера."""

    def test_trainer_creation(self):
        """Тест создания тренера."""
        trainer = BarcodeTrainer()
        assert trainer is not None

    def test_setup_callbacks(self):
        """Тест настройки колбэков."""
        trainer = BarcodeTrainer()

        # Создаем минимальную конфигурацию
        config = OmegaConf.create({
            "output_dir": "/tmp/test",
            "checkpoint": {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 3,
                "every_n_epochs": 1
            },
            "early_stopping": {
                "monitor": "val_loss",
                "patience": 5,
                "mode": "min"
            }
        })

        callbacks = trainer.setup_callbacks(config)
        assert len(callbacks) == 3  # Checkpoint, EarlyStopping, LRMonitor
