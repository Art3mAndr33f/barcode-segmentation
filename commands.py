"""
Центральная точка входа для всех команд MLOps проекта.
Использует Fire для создания CLI интерфейса.
"""

import fire
from pathlib import Path

from barcode_segmentation.data.preprocessing import DataPreprocessor
from barcode_segmentation.training.trainer import BarcodeTrainer
from barcode_segmentation.inference.predictor import BarcodePredictor
from barcode_segmentation.deployment.onnx_converter import ONNXConverter
from barcode_segmentation.deployment.tensorrt_converter import TensorRTConverter


class BarcodeSegmentationCLI:
    """Главный CLI класс для управления проектом сегментации штрих-кодов."""

    def __init__(self):
        """Инициализация CLI."""
        self.project_root = Path(__file__).parent

    def preprocess(self, config_path: str = "configs/data/preprocessing.yaml"):
        """
        Предобработка данных.

        Args:
            config_path: Путь к конфигурационному файлу
        """
        preprocessor = DataPreprocessor()
        preprocessor.run(config_path)

    def train(self, config_path: str = "configs/train.yaml"):
        """
        Запуск тренировки модели.

        Args:
            config_path: Путь к конфигурационному файлу для тренировки
        """
        trainer = BarcodeTrainer()
        trainer.run(config_path)

    def infer(self, config_path: str = "configs/inference.yaml"):
        """
        Запуск инференса модели.

        Args:
            config_path: Путь к конфигурационному файлу для инференса
        """
        predictor = BarcodePredictor()
        predictor.run(config_path)

    def convert_to_onnx(self, config_path: str = "configs/deployment.yaml"):
        """
        Конвертация модели в ONNX формат.

        Args:
            config_path: Путь к конфигурационному файлу для деплоймента
        """
        converter = ONNXConverter()
        converter.run(config_path)

    def convert_to_tensorrt(self, config_path: str = "configs/deployment.yaml"):
        """
        Конвертация модели в TensorRT формат.

        Args:
            config_path: Путь к конфигурационному файлу для деплоймента
        """
        converter = TensorRTConverter()
        converter.run(config_path)

    def setup_dvc(self, remote_url: str = None):
        """
        Настройка DVC для управления данными.

        Args:
            remote_url: URL удаленного хранилища (опционально)
        """
        import subprocess

        # Инициализируем DVC
        subprocess.run(["dvc", "init"], cwd=self.project_root)

        # Добавляем данные под контроль DVC
        subprocess.run(["dvc", "add", "data/raw"], cwd=self.project_root)
        subprocess.run(["dvc", "add", "models"], cwd=self.project_root)

        if remote_url:
            subprocess.run(["dvc", "remote", "add", "-d", "storage", remote_url], 
                         cwd=self.project_root)

        print("DVC настроен успешно!")

    def setup_mlflow(self, tracking_uri: str = "http://127.0.0.1:8080"):
        """
        Настройка MLflow для отслеживания экспериментов.

        Args:
            tracking_uri: URI для MLflow tracking server
        """
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow настроен с tracking URI: {tracking_uri}")


def main():
    """Главная функция для запуска CLI."""
    fire.Fire(BarcodeSegmentationCLI)


if __name__ == "__main__":
    main()
