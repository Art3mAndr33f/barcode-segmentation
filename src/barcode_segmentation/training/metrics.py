#!/usr/bin/env python3
"""
Модуль для вычисления метрик сегментации штрих-кодов.

Содержит различные метрики для оценки качества сегментации,
включая стандартные метрики компьютерного зрения и 
специализированные метрики для штрих-кодов.
"""

import logging
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BarcodeMetrics:
    """
    Класс для вычисления метрик сегментации штрих-кодов.
    
    Содержит методы для вычисления различных метрик качества:
    - IoU (Intersection over Union)
    - Modified IoU для штрих-кодов
    - Precision, Recall, F1-score
    - Average Precision (AP)
    - Mean Average Precision (mAP)
    """
    
    def __init__(self, iou_thresholds: List[float] = None):
        """
        Инициализация класса метрик.
        
        Args:
            iou_thresholds: Пороги IoU для вычисления AP
        """
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
    def compute_batch_metrics(
        self, 
        predictions: List[Dict[str, Any]], 
        targets: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисление метрик для батча.
        
        Args:
            predictions: Список предсказаний
            targets: Список целей
            
        Returns:
            Словарь с метриками батча
        """
        if not predictions or not targets:
            return {}
        
        batch_ious = []
        batch_modified_ious = []
        batch_precisions = []
        batch_recalls = []
        batch_f1_scores = []
        
        for pred, target in zip(predictions, targets):
            # Вычисляем метрики для одного изображения
            sample_metrics = self._compute_sample_metrics(pred, target)
            
            if sample_metrics:
                batch_ious.append(sample_metrics["iou"])
                batch_modified_ious.append(sample_metrics["modified_iou"])
                batch_precisions.append(sample_metrics["precision"])
                batch_recalls.append(sample_metrics["recall"])
                batch_f1_scores.append(sample_metrics["f1_score"])
        
        # Усредняем метрики по батчу
        return {
            "iou": np.mean(batch_ious) if batch_ious else 0.0,
            "modified_iou": np.mean(batch_modified_ious) if batch_modified_ious else 0.0,
            "precision": np.mean(batch_precisions) if batch_precisions else 0.0,
            "recall": np.mean(batch_recalls) if batch_recalls else 0.0,
            "f1_score": np.mean(batch_f1_scores) if batch_f1_scores else 0.0,
        }
    
    def compute_epoch_metrics(
        self,
        all_predictions: List[List[Dict[str, Any]]],
        all_targets: List[List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Вычисление метрик для эпохи.
        
        Args:
            all_predictions: Все предсказания эпохи
            all_targets: Все цели эпохи
            
        Returns:
            Словарь с метриками эпохи
        """
        # Собираем все IoU для вычисления mAP
        all_ious = []
        all_modified_ious = []
        all_precisions = []
        all_recalls = []
        all_f1_scores = []
        
        for predictions, targets in zip(all_predictions, all_targets):
            if not predictions or not targets:
                continue
                
            # Вычисляем метрики для изображения
            sample_metrics = self._compute_sample_metrics(predictions, targets)
            
            if sample_metrics:
                all_ious.append(sample_metrics["iou"])
                all_modified_ious.append(sample_metrics["modified_iou"])
                all_precisions.append(sample_metrics["precision"])
                all_recalls.append(sample_metrics["recall"])
                all_f1_scores.append(sample_metrics["f1_score"])
        
        # Вычисляем mAP
        map_50 = self._compute_map(all_predictions, all_targets, iou_threshold=0.5)
        map_75 = self._compute_map(all_predictions, all_targets, iou_threshold=0.75)
        map_avg = self._compute_map(all_predictions, all_targets)
        
        return {
            "mean_iou": np.mean(all_ious) if all_ious else 0.0,
            "mean_modified_iou": np.mean(all_modified_ious) if all_modified_ious else 0.0,
            "mean_precision": np.mean(all_precisions) if all_precisions else 0.0,
            "mean_recall": np.mean(all_recalls) if all_recalls else 0.0,
            "mean_f1_score": np.mean(all_f1_scores) if all_f1_scores else 0.0,
            "map_50": map_50,
            "map_75": map_75,
            "map_avg": map_avg,
        }
    
    def _compute_sample_metrics(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисление метрик для одного образца.
        
        Args:
            predictions: Предсказания для образца
            targets: Цели для образца
            
        Returns:
            Словарь с метриками образца
        """
        if not predictions or not targets:
            return {}
        
        # Извлекаем маски из предсказаний и целей
        pred_masks = self._extract_masks(predictions)
        target_masks = self._extract_masks(targets)
        
        if len(pred_masks) == 0 or len(target_masks) == 0:
            return {
                "iou": 0.0,
                "modified_iou": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
            }
        
        # Находим наилучшие соответствия между предсказаниями и целями
        best_ious = []
        best_modified_ious = []
        
        for pred_mask in pred_masks:
            max_iou = 0.0
            max_modified_iou = 0.0
            
            for target_mask in target_masks:
                iou = self._compute_mask_iou(pred_mask, target_mask)
                modified_iou = self._compute_modified_iou(pred_mask, target_mask)
                
                max_iou = max(max_iou, iou)
                max_modified_iou = max(max_modified_iou, modified_iou)
            
            best_ious.append(max_iou)
            best_modified_ious.append(max_modified_iou)
        
        # Вычисляем precision, recall, f1
        # Считаем правильным предсказание, если IoU > 0.5
        true_positives = sum(1 for iou in best_ious if iou > 0.5)
        false_positives = len(pred_masks) - true_positives
        false_negatives = len(target_masks) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "iou": np.mean(best_ious),
            "modified_iou": np.mean(best_modified_ious),
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }
    
    def _extract_masks(self, detections: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Извлечение масок из детекций.
        
        Args:
            detections: Список детекций
            
        Returns:
            Список масок
        """
        masks = []
        
        for detection in detections:
            if "mask" in detection:
                mask_data = detection["mask"]
                if "binary_mask" in mask_data:
                    masks.append(mask_data["binary_mask"])
                elif "polygon" in mask_data and mask_data["polygon"]:
                    # Преобразуем полигон в маску (требует информацию о размере изображения)
                    # Здесь упрощенная версия
                    pass
        
        return masks
    
    def _compute_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Вычисление IoU между двумя масками.
        
        Args:
            mask1: Первая маска
            mask2: Вторая маска
            
        Returns:
            IoU значение
        """
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_modified_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        Вычисление модифицированного IoU для штрих-кодов.
        
        Учитывает специфику штрих-кодов - вертикальные полосы.
        
        Args:
            mask1: Первая маска
            mask2: Вторая маска
            
        Returns:
            Модифицированное IoU значение
        """
        if mask1.shape != mask2.shape:
            return 0.0
        
        # Проецируем маски на горизонтальную ось (учитываем вертикальные полосы)
        proj1 = np.any(mask1, axis=0).astype(int)
        proj2 = np.any(mask2, axis=0).astype(int)
        
        intersection = np.logical_and(proj1, proj2).sum()
        union = np.logical_or(proj1, proj2).sum()
        
        if union == 0:
            return self._compute_mask_iou(mask1, mask2)  # Fallback к стандартному IoU
        
        horizontal_iou = intersection / union
        
        # Комбинируем с обычным IoU
        standard_iou = self._compute_mask_iou(mask1, mask2)
        
        # Взвешенная комбинация (больший вес горизонтальному IoU для штрих-кодов)
        return 0.7 * horizontal_iou + 0.3 * standard_iou
    
    def _compute_map(
        self,
        all_predictions: List[List[Dict[str, Any]]],
        all_targets: List[List[Dict[str, Any]]],
        iou_threshold: float = None
    ) -> float:
        """
        Вычисление Mean Average Precision (mAP).
        
        Args:
            all_predictions: Все предсказания
            all_targets: Все цели
            iou_threshold: Порог IoU (если None, усредняется по всем порогам)
            
        Returns:
            mAP значение
        """
        if iou_threshold is not None:
            thresholds = [iou_threshold]
        else:
            thresholds = self.iou_thresholds
        
        aps = []
        
        for threshold in thresholds:
            ap = self._compute_ap_at_threshold(all_predictions, all_targets, threshold)
            aps.append(ap)
        
        return np.mean(aps)
    
    def _compute_ap_at_threshold(
        self,
        all_predictions: List[List[Dict[str, Any]]],
        all_targets: List[List[Dict[str, Any]]],
        iou_threshold: float
    ) -> float:
        """
        Вычисление AP для конкретного порога IoU.
        
        Args:
            all_predictions: Все предсказания
            all_targets: Все цели
            iou_threshold: Порог IoU
            
        Returns:
            AP значение
        """
        # Собираем все детекции с confidence scores
        detections = []
        
        for predictions in all_predictions:
            for pred in predictions:
                if "confidence" in pred:
                    detections.append({
                        "confidence": pred["confidence"],
                        "prediction": pred
                    })
        
        # Сортируем по confidence (по убыванию)
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Подсчитываем общее количество ground truth объектов
        total_gt = sum(len(targets) for targets in all_targets)
        
        if total_gt == 0:
            return 0.0
        
        # Вычисляем precision и recall для каждого порога
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        # Здесь упрощенная версия - для полной реализации нужна более сложная логика
        # сопоставления предсказаний с ground truth
        
        for detection in detections:
            # Упрощенная проверка - считаем TP, если confidence > порог
            if detection["confidence"] > iou_threshold:
                tp += 1
            else:
                fp += 1
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / total_gt if total_gt > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Вычисляем AP как площадь под кривой precision-recall
        if not precisions or not recalls:
            return 0.0
        
        # Упрощенное вычисление AP
        return np.trapz(precisions, recalls) if len(recalls) > 1 else precisions[0] if precisions else 0.0


def compute_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[str, float]:
    """
    Функция-обертка для быстрого вычисления метрик.
    
    Args:
        predictions: Список предсказаний
        targets: Список целей
        
    Returns:
        Словарь с метриками
    """
    metrics_calculator = BarcodeMetrics()
    return metrics_calculator.compute_batch_metrics(predictions, targets)


def main() -> None:
    """Основная функция для тестирования модуля."""
    # Создаем тестовые данные
    predictions = [[{
        "bbox": [10, 10, 50, 30],
        "confidence": 0.8,
        "mask": {
            "binary_mask": np.random.rand(100, 100) > 0.5
        }
    }]]
    
    targets = [[{
        "bbox": [12, 12, 48, 28],
        "mask": {
            "binary_mask": np.random.rand(100, 100) > 0.6
        }
    }]]
    
    # Вычисляем метрики
    metrics_calculator = BarcodeMetrics()
    metrics = metrics_calculator.compute_epoch_metrics(predictions, targets)
    
    print("Тестовые метрики:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()