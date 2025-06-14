#!/usr/bin/env python3
"""
Тесты для модуля метрик сегментации штрих-кодов.

Проверяет корректность вычисления различных метрик и edge cases.
"""

import unittest
from unittest.mock import Mock, patch

import numpy as np
import pytest

from barcode_segmentation.utils.metrics import BarcodeMetrics, compute_metrics


class TestBarcodeMetrics(unittest.TestCase):
    """Тесты для класса BarcodeMetrics."""

    def setUp(self):
        """Настройка тестового окружения."""
        self.metrics = BarcodeMetrics()
        
        # Тестовые данные
        self.sample_predictions = [{
            "bbox": [10, 10, 50, 30],
            "confidence": 0.8,
            "mask": {
                "binary_mask": np.ones((100, 100), dtype=bool)
            }
        }]
        
        self.sample_targets = [{
            "bbox": [12, 12, 48, 28],
            "mask": {
                "binary_mask": np.ones((100, 100), dtype=bool)
            }
        }]

    def test_initialization(self):
        """Тест инициализации класса метрик."""
        # Тест с параметрами по умолчанию
        metrics = BarcodeMetrics()
        self.assertEqual(len(metrics.iou_thresholds), 10)
        self.assertIn(0.5, metrics.iou_thresholds)
        
        # Тест с кастомными порогами
        custom_thresholds = [0.3, 0.5, 0.7]
        metrics_custom = BarcodeMetrics(iou_thresholds=custom_thresholds)
        self.assertEqual(metrics_custom.iou_thresholds, custom_thresholds)

    def test_compute_mask_iou_perfect_match(self):
        """Тест IoU для идентичных масок."""
        mask1 = np.ones((50, 50), dtype=bool)
        mask2 = np.ones((50, 50), dtype=bool)
        
        iou = self.metrics._compute_mask_iou(mask1, mask2)
        self.assertEqual(iou, 1.0)

    def test_compute_mask_iou_no_overlap(self):
        """Тест IoU для неперекрывающихся масок."""
        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[:25, :25] = True
        
        mask2 = np.zeros((50, 50), dtype=bool)
        mask2[25:, 25:] = True
        
        iou = self.metrics._compute_mask_iou(mask1, mask2)
        self.assertEqual(iou, 0.0)

    def test_compute_mask_iou_partial_overlap(self):
        """Тест IoU для частично перекрывающихся масок."""
        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[:30, :30] = True  # 900 пикселей
        
        mask2 = np.zeros((50, 50), dtype=bool)
        mask2[20:, 20:] = True  # 900 пикселей, перекрытие 300 пикселей
        
        iou = self.metrics._compute_mask_iou(mask1, mask2)
        expected_iou = 300 / (900 + 900 - 300)  # intersection / union
        self.assertAlmostEqual(iou, expected_iou, places=5)

    def test_compute_mask_iou_different_shapes(self):
        """Тест IoU для масок разного размера."""
        mask1 = np.ones((50, 50), dtype=bool)
        mask2 = np.ones((30, 30), dtype=bool)
        
        iou = self.metrics._compute_mask_iou(mask1, mask2)
        self.assertEqual(iou, 0.0)

    def test_compute_modified_iou(self):
        """Тест модифицированного IoU для штрих-кодов."""
        # Создаем вертикальные полосы (имитация штрих-кода)
        mask1 = np.zeros((50, 50), dtype=bool)
        mask1[:, 10:15] = True  # Вертикальная полоса
        mask1[:, 20:25] = True  # Еще одна вертикальная полоса
        
        mask2 = np.zeros((50, 50), dtype=bool)
        mask2[:, 12:17] = True  # Немного смещенная полоса
        mask2[:, 22:27] = True  # Еще одна смещенная полоса
        
        modified_iou = self.metrics._compute_modified_iou(mask1, mask2)
        standard_iou = self.metrics._compute_mask_iou(mask1, mask2)
        
        # Модифицированный IoU должен быть выше для штрих-кодов
        self.assertGreaterEqual(modified_iou, standard_iou)

    def test_extract_masks_with_binary_masks(self):
        """Тест извлечения бинарных масок."""
        detections = [
            {
                "mask": {
                    "binary_mask": np.ones((50, 50), dtype=bool)
                }
            },
            {
                "mask": {
                    "binary_mask": np.zeros((50, 50), dtype=bool)
                }
            }
        ]
        
        masks = self.metrics._extract_masks(detections)
        self.assertEqual(len(masks), 2)
        self.assertTrue(np.array_equal(masks[0], np.ones((50, 50), dtype=bool)))
        self.assertTrue(np.array_equal(masks[1], np.zeros((50, 50), dtype=bool)))

    def test_extract_masks_empty_detections(self):
        """Тест извлечения масок из пустого списка детекций."""
        detections = []
        masks = self.metrics._extract_masks(detections)
        self.assertEqual(len(masks), 0)

    def test_extract_masks_no_mask_data(self):
        """Тест извлечения масок из детекций без данных масок."""
        detections = [
            {"bbox": [10, 10, 50, 50]},
            {"confidence": 0.8}
        ]
        
        masks = self.metrics._extract_masks(detections)
        self.assertEqual(len(masks), 0)

    def test_compute_sample_metrics_empty_inputs(self):
        """Тест вычисления метрик для пустых входных данных."""
        # Пустые предсказания
        metrics = self.metrics._compute_sample_metrics([], self.sample_targets)
        self.assertEqual(metrics, {})
        
        # Пустые цели
        metrics = self.metrics._compute_sample_metrics(self.sample_predictions, [])
        self.assertEqual(metrics, {})
        
        # Оба пустые
        metrics = self.metrics._compute_sample_metrics([], [])
        self.assertEqual(metrics, {})

    def test_compute_batch_metrics_valid_data(self):
        """Тест вычисления метрик батча с валидными данными."""
        predictions = [self.sample_predictions]
        targets = [self.sample_targets]
        
        with patch.object(self.metrics, '_compute_sample_metrics') as mock_sample_metrics:
            mock_sample_metrics.return_value = {
                "iou": 0.8,
                "modified_iou": 0.85,
                "precision": 0.9,
                "recall": 0.95,
                "f1_score": 0.925
            }
            
            metrics = self.metrics.compute_batch_metrics(predictions, targets)
            
            self.assertIn("iou", metrics)
            self.assertIn("modified_iou", metrics)
            self.assertIn("precision", metrics)
            self.assertIn("recall", metrics)
            self.assertIn("f1_score", metrics)
            
            self.assertEqual(metrics["iou"], 0.8)
            self.assertEqual(metrics["modified_iou"], 0.85)

    def test_compute_batch_metrics_empty_inputs(self):
        """Тест вычисления метрик батча для пустых входных данных."""
        metrics = self.metrics.compute_batch_metrics([], [])
        self.assertEqual(metrics, {})

    def test_compute_epoch_metrics(self):
        """Тест вычисления метрик эпохи."""
        all_predictions = [[self.sample_predictions], [self.sample_predictions]]
        all_targets = [[self.sample_targets], [self.sample_targets]]
        
        with patch.object(self.metrics, '_compute_sample_metrics') as mock_sample_metrics:
            mock_sample_metrics.return_value = {
                "iou": 0.8,
                "modified_iou": 0.85,
                "precision": 0.9,
                "recall": 0.95,
                "f1_score": 0.925
            }
            
            with patch.object(self.metrics, '_compute_map') as mock_map:
                mock_map.return_value = 0.75
                
                metrics = self.metrics.compute_epoch_metrics(all_predictions, all_targets)
                
                self.assertIn("mean_iou", metrics)
                self.assertIn("mean_modified_iou", metrics)
                self.assertIn("map_50", metrics)
                self.assertIn("map_75", metrics)
                self.assertIn("map_avg", metrics)

    @pytest.mark.unit
    def test_metrics_consistency(self):
        """Тест консистентности метрик."""
        # Создаем детерминированные тестовые данные
        np.random.seed(42)
        
        predictions = [{
            "bbox": [10, 10, 50, 30],
            "confidence": 0.8,
            "mask": {
                "binary_mask": np.random.rand(100, 100) > 0.5
            }
        }]
        
        targets = [{
            "bbox": [12, 12, 48, 28],
            "mask": {
                "binary_mask": np.random.rand(100, 100) > 0.6
            }
        }]
        
        # Вычисляем метрики несколько раз
        metrics1 = self.metrics.compute_batch_metrics([predictions], [targets])
        metrics2 = self.metrics.compute_batch_metrics([predictions], [targets])
        
        # Результаты должны быть одинаковыми для одинаковых входных данных
        for key in metrics1:
            self.assertAlmostEqual(metrics1[key], metrics2[key], places=5)


class TestComputeMetricsFunction(unittest.TestCase):
    """Тесты для функции compute_metrics."""

    def test_compute_metrics_wrapper(self):
        """Тест функции-обертки compute_metrics."""
        predictions = [{
            "bbox": [10, 10, 50, 30],
            "confidence": 0.8,
            "mask": {
                "binary_mask": np.ones((50, 50), dtype=bool)
            }
        }]
        
        targets = [{
            "bbox": [12, 12, 48, 28],
            "mask": {
                "binary_mask": np.ones((50, 50), dtype=bool)
            }
        }]
        
        metrics = compute_metrics(predictions, targets)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn("iou", metrics)


class TestMetricsIntegration(unittest.TestCase):
    """Интеграционные тесты для модуля метрик."""

    @pytest.mark.integration
    def test_full_pipeline_metrics(self):
        """Тест полного pipeline вычисления метрик."""
        # Создаем реалистичные тестовые данные
        predictions = []
        targets = []
        
        # Создаем несколько образцов
        for i in range(5):
            # Предсказания
            pred = {
                "bbox": [10 + i, 10 + i, 50 + i, 30 + i],
                "confidence": 0.8 + i * 0.02,
                "mask": {
                    "binary_mask": np.random.rand(100, 100) > 0.4
                }
            }
            predictions.append([pred])
            
            # Цели (немного смещенные)
            target = {
                "bbox": [12 + i, 12 + i, 48 + i, 28 + i],
                "mask": {
                    "binary_mask": np.random.rand(100, 100) > 0.5
                }
            }
            targets.append([target])
        
        metrics_calculator = BarcodeMetrics()
        
        # Вычисляем метрики эпохи
        epoch_metrics = metrics_calculator.compute_epoch_metrics(predictions, targets)
        
        # Проверяем наличие всех ожидаемых метрик
        expected_metrics = [
            "mean_iou", "mean_modified_iou", "mean_precision", 
            "mean_recall", "mean_f1_score", "map_50", "map_75", "map_avg"
        ]
        
        for metric_name in expected_metrics:
            self.assertIn(metric_name, epoch_metrics)
            self.assertIsInstance(epoch_metrics[metric_name], (int, float))
            self.assertGreaterEqual(epoch_metrics[metric_name], 0.0)
            self.assertLessEqual(epoch_metrics[metric_name], 1.0)


if __name__ == "__main__":
    # Запуск тестов
    unittest.main(verbosity=2)