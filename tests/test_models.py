"""
Тесты для модуля моделей.
"""

import pytest
import torch
from omegaconf import OmegaConf

from barcode_segmentation.models.utils import ModelUtils


class TestModelUtils:
    """Тесты для утилит моделей."""

    def test_calculate_iou(self):
        """Тест вычисления IoU."""
        # Создаем тестовые маски
        mask1 = torch.zeros(100, 100, dtype=torch.bool)
        mask2 = torch.zeros(100, 100, dtype=torch.bool)

        # Полностью совпадающие маски
        mask1[20:80, 20:80] = True
        mask2[20:80, 20:80] = True

        iou = ModelUtils.calculate_iou(mask1.numpy(), mask2.numpy())
        assert iou == 1.0

        # Частично пересекающиеся маски
        mask2[40:100, 40:100] = True
        iou = ModelUtils.calculate_iou(mask1.numpy(), mask2.numpy())
        assert 0 < iou < 1

    def test_calculate_modified_iou(self):
        """Тест вычисления модифицированного IoU."""
        # Создаем тестовые маски
        predicted = torch.zeros(100, 100, dtype=torch.bool)
        ground_truth = torch.zeros(100, 100, dtype=torch.bool)

        ground_truth[20:80, 20:80] = True
        predicted[30:90, 30:90] = True

        modified_iou = ModelUtils.calculate_modified_iou(
            predicted.numpy(), 
            ground_truth.numpy()
        )

        assert isinstance(modified_iou, float)
        assert modified_iou >= 0
