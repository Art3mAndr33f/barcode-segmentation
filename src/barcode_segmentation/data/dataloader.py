"""
DataModule для PyTorch Lightning с интеграцией Detectron2.
"""

from pathlib import Path
from typing import Optional, Union

import lightning as L
from detectron2.data import DatasetCatalog, MetadataCatalog
from omegaconf import DictConfig

from .dataset import BarcodeDataset, get_barcode_dicts


class BarcodeDataModule(L.LightningDataModule):
    """
    Lightning DataModule для работы с данными штрих-кодов.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 4,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        **kwargs
    ):
        """
        Инициализация DataModule.

        Args:
            data_dir: Путь к директории с данными
            batch_size: Размер батча
            num_workers: Количество workers для DataLoader
            train_split: Доля данных для тренировки
            val_split: Доля данных для валидации
            test_split: Доля данных для тестирования
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        # Проверяем что суммы разбиений равны 1
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6,             "Сумма разбиений должна быть равна 1.0"

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Подготовка данных (загрузка, если необходимо).
        Вызывается только на главном процессе.
        """
        # Проверяем наличие данных
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Директория с данными не найдена: {self.data_dir}")

        # Проверяем наличие изображений
        image_files = list(self.data_dir.glob("*.jpg"))
        if not image_files:
            raise FileNotFoundError(f"Изображения не найдены в {self.data_dir}")

        print(f"Найдено {len(image_files)} изображений в {self.data_dir}")

    def setup(self, stage: Optional[str] = None):
        """
        Настройка датасетов для каждой стадии.

        Args:
            stage: Стадия ('fit', 'validate', 'test', 'predict')
        """
        if stage == "fit" or stage is None:
            # Загружаем полный датасет
            full_dataset = BarcodeDataset(self.data_dir)
            dataset_dicts = full_dataset.get_dataset_dicts()

            # Разбиваем на train/val/test
            total_size = len(dataset_dicts)
            train_size = int(total_size * self.train_split)
            val_size = int(total_size * self.val_split)

            train_dicts = dataset_dicts[:train_size]
            val_dicts = dataset_dicts[train_size:train_size + val_size]
            test_dicts = dataset_dicts[train_size + val_size:]

            # Регистрируем датасеты в Detectron2
            self._register_datasets(train_dicts, val_dicts, test_dicts)

            print(f"Разбиение данных:")
            print(f"  Тренировка: {len(train_dicts)} изображений")
            print(f"  Валидация: {len(val_dicts)} изображений")
            print(f"  Тест: {len(test_dicts)} изображений")

    def _register_datasets(self, train_dicts, val_dicts, test_dicts):
        """
        Регистрирует датасеты в Detectron2.

        Args:
            train_dicts: Данные для тренировки
            val_dicts: Данные для валидации  
            test_dicts: Данные для тестирования
        """
        # Регистрируем тренировочный датасет
        if "barcode_train" in DatasetCatalog:
            DatasetCatalog.remove("barcode_train")
        DatasetCatalog.register("barcode_train", lambda: train_dicts)
        MetadataCatalog.get("barcode_train").set(thing_classes=["barcode"])

        # Регистрируем валидационный датасет
        if "barcode_val" in DatasetCatalog:
            DatasetCatalog.remove("barcode_val")
        DatasetCatalog.register("barcode_val", lambda: val_dicts)
        MetadataCatalog.get("barcode_val").set(thing_classes=["barcode"])

        # Регистрируем тестовый датасет
        if "barcode_test" in DatasetCatalog:
            DatasetCatalog.remove("barcode_test")
        DatasetCatalog.register("barcode_test", lambda: test_dicts)
        MetadataCatalog.get("barcode_test").set(thing_classes=["barcode"])

    def train_dataloader(self):
        """
        Возвращает DataLoader для тренировки.
        Для Detectron2 возвращаем None, так как он использует свой DataLoader.
        """
        return None

    def val_dataloader(self):
        """
        Возвращает DataLoader для валидации.
        Для Detectron2 возвращаем None, так как он использует свой DataLoader.
        """
        return None

    def test_dataloader(self):
        """
        Возвращает DataLoader для тестирования.
        Для Detectron2 возвращаем None, так как он использует свой DataLoader.
        """
        return None

    @classmethod
    def from_config(cls, config: DictConfig) -> "BarcodeDataModule":
        """
        Создает DataModule из конфигурации Hydra.

        Args:
            config: Конфигурация Hydra

        Returns:
            Экземпляр BarcodeDataModule
        """
        return cls(
            data_dir=config.data_dir,
            batch_size=config.data_loading.batch_size,
            num_workers=config.data_loading.num_workers,
            train_split=config.dataset.train_split,
            val_split=config.dataset.val_split,
            test_split=config.dataset.test_split,
        )
