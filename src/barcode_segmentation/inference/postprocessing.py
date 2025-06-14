"""
Модуль для постобработки результатов предсказаний.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import cv2
from detectron2.structures import Instances


class PostProcessor:
    """Класс для постобработки результатов сегментации."""

    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 nms_threshold: float = 0.5,
                 min_area: int = 100):
        """
        Инициализация постпроцессора.

        Args:
            confidence_threshold: Порог уверенности
            nms_threshold: Порог для NMS
            min_area: Минимальная площадь детекции
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.min_area = min_area

    def filter_predictions(self, predictions: Instances) -> Instances:
        """
        Фильтрует предсказания по различным критериям.

        Args:
            predictions: Предсказания модели

        Returns:
            Отфильтрованные предсказания
        """
        if len(predictions) == 0:
            return predictions

        # Фильтр по уверенности
        valid_indices = predictions.scores >= self.confidence_threshold

        # Фильтр по площади
        if hasattr(predictions, 'pred_masks'):
            areas = self._calculate_mask_areas(predictions.pred_masks)
            valid_indices = valid_indices & (areas >= self.min_area)
        elif hasattr(predictions, 'pred_boxes'):
            areas = self._calculate_box_areas(predictions.pred_boxes.tensor)
            valid_indices = valid_indices & (areas >= self.min_area)

        # Применяем фильтр
        filtered_predictions = predictions[valid_indices]

        return filtered_predictions

    def _calculate_mask_areas(self, masks: np.ndarray) -> np.ndarray:
        """
        Вычисляет площади масок.

        Args:
            masks: Массив масок

        Returns:
            Массив площадей
        """
        if len(masks.shape) == 3:
            return np.sum(masks, axis=(1, 2))
        else:
            return np.array([np.sum(mask) for mask in masks])

    def _calculate_box_areas(self, boxes: np.ndarray) -> np.ndarray:
        """
        Вычисляет площади bounding boxes.

        Args:
            boxes: Массив bounding boxes в формате [x1, y1, x2, y2]

        Returns:
            Массив площадей
        """
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        return widths * heights

    def apply_nms(self, predictions: Instances) -> Instances:
        """
        Применяет Non-Maximum Suppression.

        Args:
            predictions: Предсказания модели

        Returns:
            Предсказания после NMS
        """
        if len(predictions) <= 1:
            return predictions

        # Получаем boxes и scores
        boxes = predictions.pred_boxes.tensor.cpu().numpy()
        scores = predictions.scores.cpu().numpy()

        # Применяем NMS
        indices = self._nms(boxes, scores, self.nms_threshold)

        return predictions[indices]

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """
        Реализация Non-Maximum Suppression.

        Args:
            boxes: Массив bounding boxes
            scores: Массив scores
            threshold: Порог IoU для NMS

        Returns:
            Индексы оставшихся boxes
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]

        return keep

    def extract_barcode_regions(self, 
                               image: np.ndarray,
                               predictions: Instances) -> List[np.ndarray]:
        """
        Извлекает регионы штрих-кодов из изображения.

        Args:
            image: Входное изображение
            predictions: Предсказания модели

        Returns:
            Список изображений штрих-кодов
        """
        if len(predictions) == 0:
            return []

        barcode_regions = []

        if hasattr(predictions, 'pred_masks'):
            # Используем маски для извлечения
            masks = predictions.pred_masks.cpu().numpy()
            for mask in masks:
                region = self._extract_masked_region(image, mask)
                if region is not None:
                    barcode_regions.append(region)

        elif hasattr(predictions, 'pred_boxes'):
            # Используем bounding boxes
            boxes = predictions.pred_boxes.tensor.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.astype(int)
                region = image[y1:y2, x1:x2]
                if region.size > 0:
                    barcode_regions.append(region)

        return barcode_regions

    def _extract_masked_region(self, 
                              image: np.ndarray,
                              mask: np.ndarray) -> np.ndarray:
        """
        Извлекает регион по маске.

        Args:
            image: Входное изображение
            mask: Маска

        Returns:
            Извлеченный регион
        """
        # Находим bounding box маски
        coords = np.where(mask)
        if len(coords[0]) == 0:
            return None

        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()

        # Извлекаем регион
        region = image[y_min:y_max+1, x_min:x_max+1]
        mask_region = mask[y_min:y_max+1, x_min:x_max+1]

        # Применяем маску
        if len(region.shape) == 3:
            mask_region = np.stack([mask_region] * 3, axis=2)

        region = region * mask_region

        return region

    def enhance_barcode_image(self, barcode_image: np.ndarray) -> np.ndarray:
        """
        Улучшает изображение штрих-кода для лучшего распознавания.

        Args:
            barcode_image: Изображение штрих-кода

        Returns:
            Улучшенное изображение
        """
        # Конвертируем в оттенки серого
        if len(barcode_image.shape) == 3:
            gray = cv2.cvtColor(barcode_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = barcode_image.copy()

        # Применяем фильтрацию
        enhanced = cv2.bilateralFilter(gray, 9, 75, 75)

        # Увеличиваем контраст
        enhanced = cv2.equalizeHist(enhanced)

        # Применяем морфологические операции для очистки
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        return enhanced

    def process_predictions(self, 
                           image: np.ndarray,
                           predictions: Instances) -> Dict:
        """
        Полная обработка предсказаний.

        Args:
            image: Входное изображение
            predictions: Предсказания модели

        Returns:
            Обработанные результаты
        """
        # Фильтруем предсказания
        filtered_predictions = self.filter_predictions(predictions)

        # Применяем NMS
        final_predictions = self.apply_nms(filtered_predictions)

        # Извлекаем регионы штрих-кодов
        barcode_regions = self.extract_barcode_regions(image, final_predictions)

        # Улучшаем изображения штрих-кодов
        enhanced_regions = [
            self.enhance_barcode_image(region) 
            for region in barcode_regions
        ]

        result = {
            "original_predictions": predictions,
            "filtered_predictions": filtered_predictions,
            "final_predictions": final_predictions,
            "barcode_regions": barcode_regions,
            "enhanced_regions": enhanced_regions,
            "num_barcodes": len(final_predictions)
        }

        return result
