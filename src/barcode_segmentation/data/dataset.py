"""
Модуль для работы с датасетом штрих-кодов.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from detectron2.structures import BoxMode
from torch.utils.data import Dataset


class BarcodeDataset(Dataset):
    """
    Датасет для работы с данными штрих-кодов в формате Detectron2.
    """

    def __init__(self, data_dir: Union[str, Path], transforms=None):
        """
        Инициализация датасета.

        Args:
            data_dir: Путь к директории с данными
            transforms: Трансформации для изображений
        """
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.dataset_dicts = self._load_dataset()

    def _load_dataset(self) -> List[Dict]:
        """
        Загружает данные датасета из json файлов.

        Returns:
            Список словарей с информацией об изображениях и аннотациях
        """
        image_files = list(self.data_dir.glob("*.jpg"))
        dataset_dicts = []

        for idx, img_file in enumerate(image_files):
            json_file = img_file.with_suffix(".jpg.json")

            if not json_file.exists():
                print(f"JSON файл не найден: {json_file}")
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    img_anns = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Ошибка при загрузке {json_file}: {e}")
                continue

            record = self._create_record(img_file, img_anns, idx)
            if record:
                dataset_dicts.append(record)

        return dataset_dicts

    def _create_record(self, img_file: Path, img_anns: Dict, idx: int) -> Dict:
        """
        Создает запись датасета для одного изображения.

        Args:
            img_file: Путь к файлу изображения
            img_anns: Аннотации изображения
            idx: Индекс изображения

        Returns:
            Словарь с информацией об изображении
        """
        try:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Не удалось загрузить изображение: {img_file}")
                return None

            real_height, real_width = img.shape[:2]

            # Получаем размеры из аннотации
            ann_width, ann_height = img_anns.get("size", [real_width, real_height])

            # Проверяем соответствие размеров
            if real_width != ann_width or real_height != ann_height:
                print(f"Несоответствие размеров для {img_file}. "
                      f"Аннотация: ({ann_width}, {ann_height}), "
                      f"Реальное: ({real_width}, {real_height})")
                # Используем реальные размеры
                width, height = real_width, real_height
            else:
                width, height = ann_width, ann_height

            record = {
                "file_name": str(img_file),
                "image_id": idx,
                "height": height,
                "width": width,
                "annotations": self._parse_annotations(img_anns.get("objects", []))
            }

            return record

        except Exception as e:
            print(f"Ошибка при обработке {img_file}: {e}")
            return None

    def _parse_annotations(self, objects: List[Dict]) -> List[Dict]:
        """
        Парсит аннотации объектов.

        Args:
            objects: Список объектов из аннотации

        Returns:
            Список аннотаций в формате Detectron2
        """
        annotations = []

        for obj in objects:
            try:
                poly = np.array(obj["data"]).astype(np.int32)
                px = poly[:, 0]
                py = poly[:, 1]

                # Вычисляем bounding box
                bbox = [np.min(px), np.min(py), np.max(px), np.max(py)]
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]

                # Проверяем валидность bbox
                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                # Создаем сегментационную маску
                segmentation = [poly.flatten().tolist()]

                annotation = {
                    "bbox": [bbox[0], bbox[1], bbox_width, bbox_height],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": segmentation,
                    "category_id": 0,  # Штрих-код - категория 0
                    "iscrowd": 0,
                }

                annotations.append(annotation)

            except (KeyError, ValueError, IndexError) as e:
                print(f"Ошибка при парсинге аннотации: {e}")
                continue

        return annotations

    def __len__(self) -> int:
        """Возвращает размер датасета."""
        return len(self.dataset_dicts)

    def __getitem__(self, idx: int) -> Dict:
        """
        Возвращает элемент датасета по индексу.

        Args:
            idx: Индекс элемента

        Returns:
            Словарь с данными изображения и аннотациями
        """
        record = self.dataset_dicts[idx].copy()

        # Загружаем изображение
        image = cv2.imread(record["file_name"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            # Применяем трансформации если нужно
            transformed = self.transforms(image=image)
            image = transformed["image"]

        record["image"] = image
        return record

    def get_dataset_dicts(self) -> List[Dict]:
        """
        Возвращает список всех записей датасета.
        Используется для регистрации в Detectron2.

        Returns:
            Список словарей с данными датасета
        """
        return self.dataset_dicts


def get_barcode_dicts(data_dir: Union[str, Path]) -> List[Dict]:
    """
    Функция для получения данных в формате Detectron2.

    Args:
        data_dir: Путь к директории с данными

    Returns:
        Список словарей с данными для Detectron2
    """
    dataset = BarcodeDataset(data_dir)
    return dataset.get_dataset_dicts()
