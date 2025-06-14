"""
Модуль для предобработки данных.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Union

import hydra
from omegaconf import DictConfig
import subprocess


class DataPreprocessor:
    """Класс для предобработки данных штрих-кодов."""

    def __init__(self):
        """Инициализация препроцессора."""
        pass

    def setup_dvc(self, data_dir: Union[str, Path]):
        """
        Настройка DVC для управления данными.

        Args:
            data_dir: Путь к директории с данными
        """
        data_dir = Path(data_dir)

        # Инициализируем DVC если еще не инициализирован
        if not Path(".dvc").exists():
            subprocess.run(["dvc", "init"], check=True)
            print("DVC инициализирован")

        # Добавляем данные под контроль DVC
        if data_dir.exists():
            subprocess.run(["dvc", "add", str(data_dir)], check=True)
            print(f"Данные {data_dir} добавлены под контроль DVC")

        # Добавляем в git
        subprocess.run(["git", "add", f"{data_dir}.dvc", ".gitignore"], check=True)
        print("DVC метафайлы добавлены в git")

    def validate_data_structure(self, data_dir: Union[str, Path]) -> bool:
        """
        Проверяет структуру данных.

        Args:
            data_dir: Путь к директории с данными

        Returns:
            True если структура валидна
        """
        data_dir = Path(data_dir)

        if not data_dir.exists():
            print(f"Директория {data_dir} не существует")
            return False

        # Ищем изображения
        image_files = list(data_dir.glob("*.jpg"))
        if not image_files:
            print(f"Изображения не найдены в {data_dir}")
            return False

        # Проверяем наличие JSON файлов
        missing_json = []
        for img_file in image_files:
            json_file = img_file.with_suffix(".jpg.json")
            if not json_file.exists():
                missing_json.append(json_file.name)

        if missing_json:
            print(f"Отсутствуют JSON файлы: {missing_json[:5]}")
            if len(missing_json) > 5:
                print(f"... и еще {len(missing_json) - 5} файлов")
            return False

        print(f"Найдено {len(image_files)} пар изображение-аннотация")
        return True

    def split_data(self, 
                   data_dir: Union[str, Path],
                   output_dir: Union[str, Path],
                   train_split: float = 0.8,
                   val_split: float = 0.1,
                   test_split: float = 0.1) -> Tuple[int, int, int]:
        """
        Разбивает данные на train/val/test.

        Args:
            data_dir: Исходная директория с данными
            output_dir: Выходная директория
            train_split: Доля тренировочных данных
            val_split: Доля валидационных данных  
            test_split: Доля тестовых данных

        Returns:
            Количество файлов в каждом разбиении
        """
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)

        # Проверяем сумму разбиений
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6,             "Сумма разбиений должна быть равна 1.0"

        # Создаем выходные директории
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"  
        test_dir = output_dir / "test"

        for split_dir in [train_dir, val_dir, test_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)

        # Получаем список файлов
        image_files = sorted(list(data_dir.glob("*.jpg")))

        # Вычисляем размеры разбиений
        total_files = len(image_files)
        train_size = int(total_files * train_split)
        val_size = int(total_files * val_split)
        test_size = total_files - train_size - val_size

        # Разбиваем файлы
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]

        # Копируем файлы
        self._copy_files(train_files, train_dir)
        self._copy_files(val_files, val_dir)
        self._copy_files(test_files, test_dir)

        print(f"Данные разбиты:")
        print(f"  Тренировка: {len(train_files)} файлов")
        print(f"  Валидация: {len(val_files)} файлов")
        print(f"  Тест: {len(test_files)} файлов")

        return len(train_files), len(val_files), len(test_files)

    def _copy_files(self, files: List[Path], target_dir: Path):
        """
        Копирует файлы в целевую директорию.

        Args:
            files: Список файлов для копирования
            target_dir: Целевая директория
        """
        for img_file in files:
            # Копируем изображение
            shutil.copy2(img_file, target_dir)

            # Копируем соответствующий JSON
            json_file = img_file.with_suffix(".jpg.json")
            if json_file.exists():
                shutil.copy2(json_file, target_dir)

    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def run(self, config: DictConfig):
        """
        Запуск предобработки с конфигурацией Hydra.

        Args:
            config: Конфигурация Hydra
        """
        print("Запуск предобработки данных...")

        # Проверяем структуру данных
        raw_data_dir = Path(config.data_dir) / "raw"
        if not self.validate_data_structure(raw_data_dir):
            raise ValueError("Неверная структура данных")

        # Настраиваем DVC
        self.setup_dvc(raw_data_dir)

        # Разбиваем данные
        processed_dir = Path(config.data_dir) / "processed"
        self.split_data(
            raw_data_dir,
            processed_dir,
            config.dataset.train_split,
            config.dataset.val_split,
            config.dataset.test_split
        )

        print("Предобработка завершена успешно!")


def download_data():
    """
    Функция для загрузки данных из открытых источников.
    Используется когда нет возможности загрузить через DVC.
    """
    print("Функция загрузки данных")
    print("Поскольку данные находятся на локальной машине,")
    print("скопируйте их в директорию data/raw/")
    print("Структура должна быть:")
    print("  data/raw/")
    print("    ├── image1.jpg")
    print("    ├── image1.jpg.json")
    print("    ├── image2.jpg")
    print("    ├── image2.jpg.json")
    print("    └── ...")


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.run()
