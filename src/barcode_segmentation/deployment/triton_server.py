"""
Модуль для настройки Triton Inference Server.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Union

import hydra
from omegaconf import DictConfig


class TritonServer:
    """Класс для настройки и работы с Triton Inference Server."""

    def __init__(self):
        """Инициализация Triton сервера."""
        pass

    def create_model_repository(self, 
                               repository_path: Union[str, Path],
                               model_name: str,
                               model_version: int = 1) -> Path:
        """
        Создает структуру репозитория моделей для Triton.

        Args:
            repository_path: Путь к репозиторию моделей
            model_name: Имя модели
            model_version: Версия модели

        Returns:
            Путь к созданной модели
        """
        repo_path = Path(repository_path)
        model_path = repo_path / model_name
        version_path = model_path / str(model_version)

        # Создаем директории
        version_path.mkdir(parents=True, exist_ok=True)

        print(f"Создан репозиторий модели: {model_path}")
        return model_path

    def create_onnx_model_config(self,
                                model_name: str,
                                input_shape: List[int] = [3, 640, 640],
                                output_names: List[str] = ["output"],
                                max_batch_size: int = 8) -> Dict:
        """
        Создает конфигурацию для ONNX модели в Triton.

        Args:
            model_name: Имя модели
            input_shape: Форма входных данных
            output_names: Имена выходных тензоров
            max_batch_size: Максимальный размер батча

        Returns:
            Конфигурация модели
        """
        config = {
            "name": model_name,
            "platform": "onnxruntime_onnx",
            "max_batch_size": max_batch_size,
            "input": [
                {
                    "name": "input",
                    "data_type": "TYPE_FP32",
                    "dims": input_shape
                }
            ],
            "output": []
        }

        # Добавляем выходы (для Detectron2 это может быть сложнее)
        for output_name in output_names:
            config["output"].append({
                "name": output_name,
                "data_type": "TYPE_FP32",
                "dims": [-1]  # Динамический размер
            })

        return config

    def deploy_onnx_model(self,
                         onnx_model_path: Union[str, Path],
                         repository_path: Union[str, Path],
                         model_name: str,
                         model_version: int = 1,
                         config_override: Dict = None) -> bool:
        """
        Развертывает ONNX модель в Triton репозитории.

        Args:
            onnx_model_path: Путь к ONNX модели
            repository_path: Путь к репозиторию Triton
            model_name: Имя модели
            model_version: Версия модели
            config_override: Переопределение конфигурации

        Returns:
            True если развертывание успешно
        """
        try:
            # Создаем структуру репозитория
            model_path = self.create_model_repository(
                repository_path, model_name, model_version
            )

            # Копируем модель
            version_path = model_path / str(model_version)
            target_model_path = version_path / "model.onnx"
            shutil.copy2(onnx_model_path, target_model_path)

            # Создаем конфигурацию
            if config_override:
                config = config_override
            else:
                config = self.create_onnx_model_config(model_name)

            # Сохраняем конфигурацию
            config_path = model_path / "config.pbtxt"
            self._save_config_pbtxt(config, config_path)

            print(f"ONNX модель развернута: {model_path}")
            return True

        except Exception as e:
            print(f"Ошибка при развертывании ONNX модели: {e}")
            return False

    def _save_config_pbtxt(self, config: Dict, config_path: Path):
        """
        Сохраняет конфигурацию в формате .pbtxt.

        Args:
            config: Словарь конфигурации
            config_path: Путь для сохранения
        """
        lines = []
        lines.append(f'name: "{config["name"]}"')
        lines.append(f'platform: "{config["platform"]}"')
        lines.append(f'max_batch_size: {config["max_batch_size"]}')

        # Входы
        for inp in config["input"]:
            lines.append("input [")
            lines.append("  {")
            lines.append(f'    name: "{inp["name"]}"')
            lines.append(f'    data_type: {inp["data_type"]}')
            dims_str = ", ".join(str(d) for d in inp["dims"])
            lines.append(f'    dims: [ {dims_str} ]')
            lines.append("  }")
            lines.append("]")

        # Выходы
        for out in config["output"]:
            lines.append("output [")
            lines.append("  {")
            lines.append(f'    name: "{out["name"]}"')
            lines.append(f'    data_type: {out["data_type"]}')
            dims_str = ", ".join(str(d) for d in out["dims"])
            lines.append(f'    dims: [ {dims_str} ]')
            lines.append("  }")
            lines.append("]")

        with open(config_path, 'w') as f:
            f.write("\n".join(lines))

    def create_client_example(self, output_path: Union[str, Path], model_name: str):
        """
        Создает пример клиента для Triton сервера.

        Args:
            output_path: Путь для сохранения примера
            model_name: Имя модели
        """
        # Создаем содержимое файла как список строк
        client_lines = [
            '"""',
            'Пример клиента для Triton Inference Server.',
            '"""',
            '',
            'import numpy as np',
            'import tritonclient.http as httpclient',
            'from tritonclient.utils import InferenceServerException',
            'import cv2',
            '',
            '',
            'class TritonClient:',
            '    """Клиент для работы с Triton Inference Server."""',
            '    ',
            '    def __init__(self, server_url: str = "localhost:8000"):',
            '        """',
            '        Инициализация клиента.',
            '        ',
            '        Args:',
            '            server_url: URL сервера',
            '        """',
            '        self.server_url = server_url',
            '        self.client = httpclient.InferenceServerClient(url=server_url)',
            f'        self.model_name = "{model_name}"',
            '    ',
            '    def predict(self, image_path: str):',
            '        """Выполняет предсказание через Triton сервер."""',
            '        # Здесь будет код предсказания',
            '        pass'
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(client_lines))

        print(f"Пример клиента создан: {output_path}")

    @hydra.main(version_base=None, config_path="../../configs", config_name="deployment")
    def run(self, config: DictConfig):
        """
        Запуск развертывания на Triton сервере.

        Args:
            config: Конфигурация Hydra
        """
        print("Настройка Triton Inference Server...")

        repository_path = Path(config.deployment.triton.model_repository)
        model_name = config.deployment.triton.model_name
        model_version = config.deployment.triton.model_version

        # Создаем репозиторий
        repository_path.mkdir(parents=True, exist_ok=True)

        # Развертываем ONNX модель если есть
        onnx_path = Path(config.deployment.onnx.output_path)
        if onnx_path.exists():
            onnx_success = self.deploy_onnx_model(
                onnx_path, 
                repository_path, 
                f"{model_name}_onnx", 
                model_version
            )
            if onnx_success:
                print("ONNX модель развернута в Triton")

        # Создаем пример клиента
        client_path = repository_path.parent / "triton_client_example.py"
        self.create_client_example(client_path, model_name)

        print(f"Triton настроен! Репозиторий: {repository_path}")
        print("Для запуска сервера используйте:")
        print(f"docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 "
              f"-v{repository_path.absolute()}:/models "
              f"nvcr.io/nvidia/tritonserver:23.10-py3 "
              f"tritonserver --model-repository=/models")


def main():
    """Точка входа для настройки Triton сервера."""
    server = TritonServer()
    server.run()


if __name__ == "__main__":
    main()
