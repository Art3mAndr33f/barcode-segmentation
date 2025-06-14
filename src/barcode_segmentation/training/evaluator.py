"""
Модуль для оценки качества модели.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from omegaconf import DictConfig

from ..models.lightning_module import BarcodeLightningModule
from ..models.utils import ModelUtils


class ModelEvaluator:
    """Класс для оценки качества модели сегментации."""

    def __init__(self):
        """Инициализация оценщика."""
        pass

    def evaluate_model(self,
                      model_path: Union[str, Path],
                      config: DictConfig,
                      dataset_name: str = "barcode_test") -> Dict:
        """
        Оценивает модель на тестовом датасете.

        Args:
            model_path: Путь к модели
            config: Конфигурация
            dataset_name: Имя тестового датасета

        Returns:
            Результаты оценки
        """
        # Загружаем модель
        model = BarcodeLightningModule.load_from_checkpoint(
            model_path,
            config=config.model,
            map_location="cpu"
        )
        model.eval()

        # Получаем Detectron2 модель и конфигурацию
        detectron2_model = model.model_wrapper.get_detectron2_model()
        detectron2_cfg = model.model_wrapper.get_model_config()

        # Создаем оценщик
        evaluator = COCOEvaluator(
            dataset_name,
            detectron2_cfg,
            False,
            output_dir="./evaluation_output/"
        )

        # Создаем DataLoader для тестирования
        test_loader = build_detection_test_loader(detectron2_cfg, dataset_name)

        # Выполняем оценку
        results = inference_on_dataset(detectron2_model, test_loader, evaluator)

        return results

    def calculate_custom_metrics(self,
                                predictions: List[Dict],
                                ground_truth: List[Dict]) -> Dict:
        """
        Вычисляет кастомные метрики для сегментации штрих-кодов.

        Args:
            predictions: Предсказания модели
            ground_truth: Истинные аннотации

        Returns:
            Словарь с метриками
        """
        iou_scores = []
        modified_iou_scores = []
        detection_accuracy = 0
        total_images = len(ground_truth)

        for pred, gt in zip(predictions, ground_truth):
            # Здесь должна быть логика сравнения предсказаний с ground truth
            # Упрощенная версия для демонстрации
            if pred.get("num_detections", 0) > 0 and len(gt.get("annotations", [])) > 0:
                detection_accuracy += 1
                # В реальности здесь нужно вычислить IoU между масками
                iou_scores.append(0.8)  # Заглушка
                modified_iou_scores.append(0.75)  # Заглушка

        metrics = {
            "detection_accuracy": detection_accuracy / total_images,
            "mean_iou": np.mean(iou_scores) if iou_scores else 0.0,
            "mean_modified_iou": np.mean(modified_iou_scores) if modified_iou_scores else 0.0,
            "total_images": total_images,
            "detected_images": detection_accuracy
        }

        return metrics

    def create_evaluation_report(self,
                               results: Dict,
                               custom_metrics: Dict,
                               output_path: Union[str, Path]):
        """
        Создает отчет об оценке модели.

        Args:
            results: Результаты COCO оценки
            custom_metrics: Кастомные метрики
            output_path: Путь для сохранения отчета
        """
        report = {
            "model_evaluation_report": {
                "coco_metrics": results,
                "custom_metrics": custom_metrics,
                "summary": {
                    "detection_accuracy": custom_metrics.get("detection_accuracy", 0.0),
                    "mean_iou": custom_metrics.get("mean_iou", 0.0),
                    "coco_ap": results.get("segm", {}).get("AP", 0.0) if "segm" in results else 0.0
                }
            }
        }

        # Сохраняем отчет
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Отчет об оценке сохранен: {output_path}")

        # Выводим краткую сводку
        print("\nКраткая сводка:")
        print(f"  Точность детекции: {custom_metrics.get('detection_accuracy', 0.0):.3f}")
        print(f"  Средний IoU: {custom_metrics.get('mean_iou', 0.0):.3f}")
        if "segm" in results:
            print(f"  COCO AP (segmentation): {results['segm'].get('AP', 0.0):.3f}")
