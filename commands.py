#!/usr/bin/env python3
"""
Центральная точка входа для всех команд MLOps проекта.
Использует Fire для создания CLI интерфейса и поддерживает все основные операции проекта.

Примеры использования:
    python commands.py preprocess --config_path configs/data/preprocessing.yaml
    python commands.py train --config_path configs/train.yaml
    python commands.py infer --config_path configs/inference.yaml
    python commands.py setup_dvc --remote_url s3://my-bucket/data
    python commands.py serve --port 8000
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import fire

from barcode_segmentation.data.preprocessing import DataPreprocessor
from barcode_segmentation.deployment.inference_server import InferenceServer
from barcode_segmentation.deployment.onnx_converter import ONNXConverter
from barcode_segmentation.deployment.tensorrt_converter import TensorRTConverter
from barcode_segmentation.inference.predictor import BarcodePredictor
from barcode_segmentation.training.trainer import BarcodeTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class BarcodeSegmentationCLI:
    """
    Главный CLI класс для управления проектом сегментации штрих-кодов.
    
    Предоставляет команды для всех этапов MLOps pipeline:
    - Предобработка данных
    - Обучение моделей
    - Инференс
    - Конвертация моделей
    - Деплоймент
    - Настройка инфраструктуры
    """

    def __init__(self):
        """Инициализация CLI с проверкой окружения."""
        self.project_root = Path(__file__).parent.resolve()
        logger.info(f"Инициализирован CLI в директории: {self.project_root}")
        
        # Проверяем структуру проекта
        self._validate_project_structure()

    def _validate_project_structure(self) -> None:
        """Проверяет наличие основных директорий проекта."""
        required_dirs = ["src", "configs", "data"]
        missing_dirs = []
        
        for dir_name in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                missing_dirs.append(dir_name)
                
        if missing_dirs:
            logger.warning(f"Отсутствуют директории: {missing_dirs}")
        else:
            logger.info("✓ Структура проекта валидна")

    def preprocess(self, config_path: str = "configs/data/preprocessing.yaml") -> None:
        """
        Предобработка данных с валидацией и аугментацией.

        Args:
            config_path: Путь к конфигурационному файлу для предобработки
        """
        logger.info(f"🔄 Запуск предобработки данных с конфигом: {config_path}")
        
        try:
            preprocessor = DataPreprocessor()
            preprocessor.run(config_path)
            logger.info("✅ Предобработка данных завершена успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка при предобработке данных: {e}")
            raise

    def train(self, config_path: str = "configs/train.yaml") -> None:
        """
        Запуск обучения модели с логированием в MLflow.

        Args:
            config_path: Путь к конфигурационному файлу для обучения
        """
        logger.info(f"🚀 Запуск обучения модели с конфигом: {config_path}")
        
        try:
            trainer = BarcodeTrainer()
            trainer.run(config_path)
            logger.info("✅ Обучение модели завершено успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка при обучении модели: {e}")
            raise

    def infer(self, config_path: str = "configs/inference.yaml") -> None:
        """
        Запуск инференса модели на новых данных.

        Args:
            config_path: Путь к конфигурационному файлу для инференса
        """
        logger.info(f"🔮 Запуск инференса модели с конфигом: {config_path}")
        
        try:
            predictor = BarcodePredictor()
            predictor.run(config_path)
            logger.info("✅ Инференс завершен успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка при инференсе: {e}")
            raise

    def convert_to_onnx(self, config_path: str = "configs/deployment.yaml") -> None:
        """
        Конвертация обученной модели в ONNX формат для кроссплатформенного деплоймента.

        Args:
            config_path: Путь к конфигурационному файлу для деплоймента
        """
        logger.info(f"📦 Конвертация модели в ONNX с конфигом: {config_path}")
        
        try:
            converter = ONNXConverter()
            converter.run(config_path)
            logger.info("✅ Конвертация в ONNX завершена успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка при конвертации в ONNX: {e}")
            raise

    def convert_to_tensorrt(self, config_path: str = "configs/deployment.yaml") -> None:
        """
        Конвертация модели в TensorRT формат для оптимизации на NVIDIA GPU.

        Args:
            config_path: Путь к конфигурационному файлу для деплоймента
        """
        logger.info(f"⚡ Конвертация модели в TensorRT с конфигом: {config_path}")
        
        try:
            converter = TensorRTConverter()
            converter.run(config_path)
            logger.info("✅ Конвертация в TensorRT завершена успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка при конвертации в TensorRT: {e}")
            raise

    def serve(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        model_path: Optional[str] = None,
        config_path: str = "configs/inference.yaml"
    ) -> None:
        """
        Запуск inference сервера для API доступа к модели.

        Args:
            host: Хост для сервера
            port: Порт для сервера
            model_path: Путь к модели (опционально)
            config_path: Путь к конфигурационному файлу
        """
        logger.info(f"🌐 Запуск inference сервера на {host}:{port}")
        
        try:
            server = InferenceServer(
                host=host,
                port=port,
                model_path=model_path,
                config_path=config_path
            )
            server.run()
        except Exception as e:
            logger.error(f"❌ Ошибка при запуске сервера: {e}")
            raise

    def setup_dvc(self, remote_url: Optional[str] = None) -> None:
        """
        Настройка DVC для управления данными и версионирования.

        Args:
            remote_url: URL удаленного хранилища (S3, GCS, Azure, etc.)
        """
        logger.info("⚙️ Настройка DVC для управления данными")
        
        try:
            # Инициализируем DVC
            result = subprocess.run(
                ["dvc", "init"], 
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✓ DVC инициализирован")
            else:
                logger.warning(f"DVC уже инициализирован или произошла ошибка: {result.stderr}")

            # Добавляем данные под контроль DVC
            data_dirs = ["data/raw", "models"]
            for data_dir in data_dirs:
                if (self.project_root / data_dir).exists():
                    subprocess.run(["dvc", "add", data_dir], cwd=self.project_root)
                    logger.info(f"✓ Добавлена директория {data_dir} под контроль DVC")

            # Настраиваем удаленное хранилище
            if remote_url:
                subprocess.run(
                    ["dvc", "remote", "add", "-d", "storage", remote_url], 
                    cwd=self.project_root
                )
                logger.info(f"✓ Настроено удаленное хранилище: {remote_url}")

            logger.info("✅ DVC настроен успешно!")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Ошибка при настройке DVC: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Неожиданная ошибка при настройке DVC: {e}")
            raise

    def setup_mlflow(self, tracking_uri: str = "http://127.0.0.1:8080") -> None:
        """
        Настройка MLflow для отслеживания экспериментов.

        Args:
            tracking_uri: URI для MLflow tracking server
        """
        logger.info(f"📊 Настройка MLflow с tracking URI: {tracking_uri}")
        
        try:
            import mlflow
            
            mlflow.set_tracking_uri(tracking_uri)
            
            # Проверяем подключение
            client = mlflow.tracking.MlflowClient(tracking_uri)
            experiments = client.search_experiments()
            
            logger.info(f"✓ MLflow настроен успешно")
            logger.info(f"✓ Найдено экспериментов: {len(experiments)}")
            logger.info(f"✅ MLflow доступен по адресу: {tracking_uri}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при настройке MLflow: {e}")
            raise

    def setup_environment(self) -> None:
        """Настройка полного окружения для проекта."""
        logger.info("🔧 Настройка полного окружения проекта")
        
        try:
            # Устанавливаем pre-commit hooks
            subprocess.run(["pre-commit", "install"], cwd=self.project_root, check=True)
            logger.info("✓ Pre-commit hooks установлены")
            
            # Создаем необходимые директории
            dirs_to_create = [
                "data/raw", "data/processed", "data/interim",
                "models", "outputs", "plots", "logs"
            ]
            
            for dir_path in dirs_to_create:
                (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
                
            logger.info("✓ Созданы необходимые директории")
            
            # Настройка базовых компонентов
            self.setup_dvc()
            self.setup_mlflow()
            
            logger.info("✅ Окружение настроено успешно!")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при настройке окружения: {e}")
            raise

    def health_check(self) -> None:
        """Проверка здоровья всех компонентов системы."""
        logger.info("🏥 Проверка здоровья системы")
        
        checks = {
            "Project Structure": self._check_project_structure,
            "Dependencies": self._check_dependencies,
            "DVC": self._check_dvc,
            "MLflow": self._check_mlflow,
            "Data": self._check_data,
        }
        
        results = {}
        for check_name, check_func in checks.items():
            try:
                results[check_name] = check_func()
                logger.info(f"✓ {check_name}: OK")
            except Exception as e:
                results[check_name] = False
                logger.error(f"❌ {check_name}: {e}")
        
        # Сводка
        passed = sum(results.values())
        total = len(results)
        logger.info(f"📋 Сводка проверки: {passed}/{total} тестов пройдено")
        
        if passed == total:
            logger.info("✅ Все проверки пройдены успешно!")
        else:
            logger.warning("⚠️ Некоторые проверки не пройдены")

    def _check_project_structure(self) -> bool:
        """Проверка структуры проекта."""
        required_paths = [
            "src/barcode_segmentation",
            "configs",
            "pyproject.toml",
            "README.md"
        ]
        return all((self.project_root / path).exists() for path in required_paths)

    def _check_dependencies(self) -> bool:
        """Проверка установленных зависимостей."""
        try:
            import torch
            import lightning
            import hydra
            import mlflow
            return True
        except ImportError:
            return False

    def _check_dvc(self) -> bool:
        """Проверка настройки DVC."""
        return (self.project_root / ".dvc").exists()

    def _check_mlflow(self) -> bool:
        """Проверка доступности MLflow."""
        try:
            import mlflow
            mlflow.get_tracking_uri()
            return True
        except Exception:
            return False

    def _check_data(self) -> bool:
        """Проверка наличия данных."""
        data_dir = self.project_root / "data" / "raw"
        return data_dir.exists() and any(data_dir.iterdir())


def main() -> None:
    """
    Главная функция для запуска CLI.
    
    Использует Fire для автоматического создания CLI интерфейса
    из класса BarcodeSegmentationCLI.
    """
    try:
        fire.Fire(BarcodeSegmentationCLI)
    except KeyboardInterrupt:
        logger.info("🛑 Операция прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()