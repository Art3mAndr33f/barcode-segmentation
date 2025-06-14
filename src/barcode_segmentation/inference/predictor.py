"""
Модуль для инференса модели сегментации штрих-кодов.
"""

import json
from pathlib import Path
from typing import Dict, List, Union

import cv2
import hydra
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from omegaconf import DictConfig

from ..models.lightning_module import BarcodeLightningModule
from ..models.utils import ModelUtils


class BarcodePredictor:
    """Класс для инференса модели сегментации штрих-кодов."""

    def __init__(self):
        """Инициализация предиктора."""
        self.model = None
        self.predictor = None
        self.device = None

    def load_model(self, model_path: Union[str, Path], config: DictConfig):
        """
        Загружает обученную модель.

        Args:
            model_path: Путь к файлу модели
            config: Конфигурация модели
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        print(f"Загружаем модель из {model_path}")

        # Определяем устройство
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Загружаем Lightning модель
        self.model = BarcodeLightningModule.load_from_checkpoint(
            model_path, 
            config=config.model,
            map_location=self.device
        )
        self.model.eval()

        # Настраиваем Detectron2 predictor
        self._setup_detectron2_predictor(config)

        print(f"Модель загружена на устройство: {self.device}")

    def _setup_detectron2_predictor(self, config: DictConfig):
        """
        Настраивает Detectron2 predictor для инференса.

        Args:
            config: Конфигурация
        """
        # Получаем конфигурацию Detectron2 из модели
        detectron2_model = self.model.model_wrapper.get_detectron2_model()
        cfg = self.model.model_wrapper.get_model_config()

        # Обновляем конфигурацию для инференса
        cfg.MODEL.WEIGHTS = ""  # Веса уже загружены
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.inference.confidence_threshold
        cfg.MODEL.DEVICE = str(self.device)

        # Создаем predictor
        self.predictor = DefaultPredictor(cfg)

        # Заменяем модель на нашу обученную
        self.predictor.model = detectron2_model

    def predict_single_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Выполняет предсказание для одного изображения.

        Args:
            image_path: Путь к изображению

        Returns:
            Результат предсказания
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Изображение не найдено: {image_path}")

        # Загружаем изображение
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        # Выполняем предсказание
        with torch.no_grad():
            outputs = self.predictor(image)

        # Форматируем результат
        result = {
            "image_path": str(image_path),
            "image_shape": image.shape,
            "predictions": outputs["instances"],
            "num_detections": len(outputs["instances"])
        }

        return result

    def predict_batch(self, 
                     image_paths: List[Union[str, Path]],
                     batch_size: int = 1) -> List[Dict]:
        """
        Выполняет предсказание для батча изображений.

        Args:
            image_paths: Список путей к изображениям
            batch_size: Размер батча (для Detectron2 обычно 1)

        Returns:
            Список результатов предсказаний
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            for image_path in batch_paths:
                try:
                    result = self.predict_single_image(image_path)
                    results.append(result)
                    print(f"Обработано: {image_path}")
                except Exception as e:
                    print(f"Ошибка при обработке {image_path}: {e}")
                    results.append({
                        "image_path": str(image_path),
                        "error": str(e),
                        "num_detections": 0
                    })

        return results

    def predict_directory(self, 
                         input_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         save_visualizations: bool = True) -> List[Dict]:
        """
        Выполняет предсказание для всех изображений в директории.

        Args:
            input_dir: Входная директория с изображениями
            output_dir: Выходная директория для результатов
            save_visualizations: Сохранять ли визуализации

        Returns:
            Список результатов предсказаний
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Находим все изображения
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(input_dir.glob(f"*{ext}"))
            image_paths.extend(input_dir.glob(f"*{ext.upper()}"))

        if not image_paths:
            raise FileNotFoundError(f"Изображения не найдены в {input_dir}")

        print(f"Найдено {len(image_paths)} изображений для обработки")

        # Выполняем предсказания
        results = self.predict_batch(image_paths)

        # Сохраняем результаты
        self._save_results(results, output_dir, save_visualizations)

        return results

    def _save_results(self, 
                     results: List[Dict],
                     output_dir: Path,
                     save_visualizations: bool):
        """
        Сохраняет результаты предсказаний.

        Args:
            results: Результаты предсказаний
            output_dir: Выходная директория
            save_visualizations: Сохранять ли визуализации
        """
        # Создаем директории
        predictions_dir = output_dir / "predictions"
        predictions_dir.mkdir(exist_ok=True)

        if save_visualizations:
            visualizations_dir = output_dir / "visualizations"
            visualizations_dir.mkdir(exist_ok=True)

        # Сохраняем предсказания в JSON формате
        serializable_results = []

        for i, result in enumerate(results):
            if "error" in result:
                serializable_results.append(result)
                continue

            # Конвертируем предсказания в сериализуемый формат
            predictions = result["predictions"]
            serializable_pred = ModelUtils.convert_predictions_to_serializable(predictions)

            serializable_result = {
                "image_path": result["image_path"],
                "image_shape": result["image_shape"],
                "num_detections": result["num_detections"],
                "predictions": serializable_pred
            }
            serializable_results.append(serializable_result)

            # Сохраняем индивидуальное предсказание
            pred_file = predictions_dir / f"prediction_{i:04d}.json"
            with open(pred_file, "w", encoding="utf-8") as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)

            # Создаем визуализацию если нужно
            if save_visualizations and result["num_detections"] > 0:
                self._create_visualization(result, visualizations_dir, i)

        # Сохраняем общий файл с результатами
        summary_file = output_dir / "predictions_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        # Сохраняем статистику
        ModelUtils.save_predictions_summary(results, output_dir / "statistics.json")

        print(f"Результаты сохранены в {output_dir}")

    def _create_visualization(self, result: Dict, vis_dir: Path, index: int):
        """
        Создает визуализацию предсказания.

        Args:
            result: Результат предсказания
            vis_dir: Директория для визуализаций
            index: Индекс изображения
        """
        try:
            # Загружаем изображение
            image = cv2.imread(result["image_path"])
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Создаем визуализацию
            vis_image = ModelUtils.visualize_predictions(
                image_rgb,
                result["predictions"],
                metadata_name="barcode_train"
            )

            # Сохраняем
            vis_file = vis_dir / f"visualization_{index:04d}.jpg"
            cv2.imwrite(str(vis_file), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        except Exception as e:
            print(f"Ошибка при создании визуализации {index}: {e}")

    @hydra.main(version_base=None, config_path="../../configs", config_name="inference")
    def run(self, config: DictConfig):
        """
        Запуск инференса с конфигурацией Hydra.

        Args:
            config: Конфигурация Hydra
        """
        print("Запуск инференса модели...")

        # Загружаем модель
        self.load_model(config.inference.model_path, config)

        # Выполняем предсказания
        results = self.predict_directory(
            config.inference.input_dir,
            config.inference.output_dir,
            config.inference.save_visualizations
        )

        print(f"Инференс завершен! Обработано {len(results)} изображений")

        # Выводим статистику
        successful = sum(1 for r in results if "error" not in r)
        total_detections = sum(r.get("num_detections", 0) for r in results)

        print(f"Статистика:")
        print(f"  Успешно обработано: {successful}/{len(results)}")
        print(f"  Общее количество детекций: {total_detections}")
        print(f"  Среднее количество детекций на изображение: {total_detections/len(results):.2f}")

        return results


def main():
    """Точка входа для инференса."""
    predictor = BarcodePredictor()
    predictor.run()


if __name__ == "__main__":
    main()
