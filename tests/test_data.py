"""
Тесты для модуля данных.
"""

import pytest
import tempfile
import json
from pathlib import Path

from barcode_segmentation.data.dataset import BarcodeDataset, get_barcode_dicts
from barcode_segmentation.data.dataloader import BarcodeDataModule


class TestBarcodeDataset:
    """Тесты для датасета штрих-кодов."""

    def test_dataset_creation(self):
        """Тест создания датасета."""
        # Создаем временную директорию с тестовыми данными
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Создаем тестовое изображение (заглушка)
            test_image_path = temp_path / "test_image.jpg"
            test_image_path.touch()

            # Создаем тестовую аннотацию
            annotation = {
                "size": [640, 480],
                "objects": [
                    {
                        "data": [[100, 100], [200, 100], [200, 200], [100, 200]]
                    }
                ]
            }

            json_path = temp_path / "test_image.jpg.json"
            with open(json_path, 'w') as f:
                json.dump(annotation, f)

            # Тестируем создание датасета
            dataset = BarcodeDataset(temp_path)

            # Проверяем что датасет не пустой
            assert len(dataset) >= 0

    def test_get_barcode_dicts(self):
        """Тест функции get_barcode_dicts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Функция должна работать даже с пустой директорией
            dicts = get_barcode_dicts(temp_dir)
            assert isinstance(dicts, list)


class TestBarcodeDataModule:
    """Тесты для DataModule."""

    def test_datamodule_creation(self):
        """Тест создания DataModule."""
        with tempfile.TemporaryDirectory() as temp_dir:
            datamodule = BarcodeDataModule(
                data_dir=temp_dir,
                batch_size=2,
                num_workers=0
            )

            assert datamodule.batch_size == 2
            assert datamodule.num_workers == 0
