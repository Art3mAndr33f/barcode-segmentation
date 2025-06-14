"""
Утилиты для работы с моделями.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from detectron2.structures import Instances
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import matplotlib.pyplot as plt


class ModelUtils:
    """Класс с утилитами для работы с моделями."""

    @staticmethod
    def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Вычисляет IoU между двумя масками.

        Args:
            mask1: Первая маска (логический массив)
            mask2: Вторая маска (логический массив)

        Returns:
            IoU значение
        """
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def calculate_modified_iou(predicted_mask: np.ndarray, 
                             ground_truth_mask: np.ndarray) -> float:
        """
        Вычисляет модифицированный IoU: f(P, T) = (S1 + 100 * S2) / ST.

        Args:
            predicted_mask: Предсказанная маска
            ground_truth_mask: Размеченная маска

        Returns:
            Модифицированный IoU
        """
        s1 = np.logical_and(predicted_mask, 
                           np.logical_not(ground_truth_mask)).sum()
        s2 = np.logical_and(ground_truth_mask, 
                           np.logical_not(predicted_mask)).sum()
        st = ground_truth_mask.sum()

        if st == 0:
            return 0.0

        return (s1 + 100 * s2) / st

    @staticmethod
    def visualize_predictions(image: np.ndarray, 
                            predictions: Instances,
                            metadata_name: str = "barcode_train",
                            save_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Визуализирует предсказания модели.

        Args:
            image: Входное изображение
            predictions: Предсказания модели
            metadata_name: Имя метаданных для визуализации
            save_path: Путь для сохранения (опционально)

        Returns:
            Визуализированное изображение
        """
        # Создаем визуализатор
        v = Visualizer(
            image[:, :, ::-1], 
            MetadataCatalog.get(metadata_name), 
            scale=1.0
        )

        # Рисуем предсказания
        out = v.draw_instance_predictions(predictions.to("cpu"))

        # Получаем изображение
        vis_image = out.get_image()[:, :, ::-1]

        # Сохраняем если указан путь
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), vis_image)

        return vis_image

    @staticmethod
    def save_predictions_summary(predictions_list: List[Dict],
                               save_path: Union[str, Path]):
        """
        Сохраняет сводку предсказаний в файл.

        Args:
            predictions_list: Список предсказаний
            save_path: Путь для сохранения
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        summary = {
            "total_images": len(predictions_list),
            "total_detections": sum(len(pred.get("instances", [])) 
                                  for pred in predictions_list),
            "average_confidence": 0.0,
            "detections_per_image": []
        }

        confidences = []
        for pred in predictions_list:
            instances = pred.get("instances", [])
            num_detections = len(instances)
            summary["detections_per_image"].append(num_detections)

            if hasattr(instances, 'scores'):
                confidences.extend(instances.scores.cpu().numpy())

        if confidences:
            summary["average_confidence"] = float(np.mean(confidences))

        # Сохраняем в JSON
        import json
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    @staticmethod
    def plot_training_metrics(train_losses: List[float],
                            val_losses: List[float],
                            save_path: Optional[Union[str, Path]] = None):
        """
        Строит графики метрик тренировки.

        Args:
            train_losses: Лоссы тренировки
            val_losses: Лоссы валидации
            save_path: Путь для сохранения графика
        """
        plt.figure(figsize=(12, 4))

        # График лоссов
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # График соотношения лоссов
        plt.subplot(1, 2, 2)
        if len(train_losses) > 0 and len(val_losses) > 0:
            ratio = [v/t if t > 0 else 1 for t, v in zip(train_losses, val_losses)]
            plt.plot(ratio, label='Val/Train Loss Ratio')
            plt.xlabel('Epoch')
            plt.ylabel('Ratio')
            plt.title('Validation/Training Loss Ratio')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    @staticmethod
    def convert_predictions_to_serializable(predictions: Instances) -> Dict:
        """
        Конвертирует предсказания в сериализуемый формат.

        Args:
            predictions: Предсказания модели

        Returns:
            Словарь с сериализуемыми данными
        """
        result = {}

        if hasattr(predictions, 'pred_boxes'):
            result['boxes'] = predictions.pred_boxes.tensor.cpu().numpy().tolist()

        if hasattr(predictions, 'scores'):
            result['scores'] = predictions.scores.cpu().numpy().tolist()

        if hasattr(predictions, 'pred_classes'):
            result['classes'] = predictions.pred_classes.cpu().numpy().tolist()

        if hasattr(predictions, 'pred_masks'):
            # Сохраняем маски как списки (сжатие возможно)
            masks = predictions.pred_masks.cpu().numpy()
            result['masks'] = masks.tolist()

        return result
