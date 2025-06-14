"""
Модуль для конвертации ONNX модели в TensorRT формат.
"""

from pathlib import Path
from typing import Union, List

import hydra
from omegaconf import DictConfig


class TensorRTConverter:
    """Класс для конвертации ONNX модели в TensorRT формат."""

    def __init__(self):
        """Инициализация конвертера."""
        self.trt_available = self._check_tensorrt_availability()

    def _check_tensorrt_availability(self) -> bool:
        """
        Проверяет доступность TensorRT.

        Returns:
            True если TensorRT доступен
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            return True
        except ImportError:
            print("TensorRT недоступен. Установите TensorRT для использования этой функции.")
            return False

    def convert_onnx_to_tensorrt(self,
                                onnx_path: Union[str, Path],
                                output_path: Union[str, Path],
                                precision: str = "fp16",
                                max_batch_size: int = 8,
                                max_workspace_size: int = 1 << 30) -> bool:
        """
        Конвертирует ONNX модель в TensorRT.

        Args:
            onnx_path: Путь к ONNX модели
            output_path: Путь для сохранения TensorRT модели
            precision: Точность (fp32, fp16, int8)
            max_batch_size: Максимальный размер батча
            max_workspace_size: Максимальный размер рабочей области

        Returns:
            True если конвертация успешна
        """
        if not self.trt_available:
            return False

        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            print(f"Конвертируем {onnx_path} в TensorRT формат...")

            # Создаем TensorRT logger
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            # Создаем builder
            builder = trt.Builder(TRT_LOGGER)

            # Создаем конфигурацию
            config = builder.create_builder_config()
            config.max_workspace_size = max_workspace_size

            # Устанавливаем точность
            if precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
                print("Используем FP16 точность")
            elif precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                print("Используем INT8 точность")

            # Создаем сеть
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )

            # Парсим ONNX
            parser = trt.OnnxParser(network, TRT_LOGGER)

            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    print("Ошибка при парсинге ONNX модели:")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return False

            print("Строим TensorRT engine...")
            engine = builder.build_engine(network, config)

            if engine is None:
                print("Не удалось создать TensorRT engine")
                return False

            # Сериализуем и сохраняем engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())

            print(f"TensorRT модель сохранена в {output_path}")
            return True

        except Exception as e:
            print(f"Ошибка при конвертации в TensorRT: {e}")
            return False

    def create_conversion_script(self, output_path: Union[str, Path]):
        """
        Создает shell скрипт для конвертации ONNX в TensorRT.

        Args:
            output_path: Путь для сохранения скрипта
        """
        script_lines = [
            "#!/bin/bash",
            "# Скрипт для конвертации ONNX модели в TensorRT",
            "",
            "# Проверяем аргументы",
            'if [ $# -ne 2 ]; then',
            '    echo "Использование: $0 <путь_к_onnx_модели> <путь_для_tensorrt_модели>"',
            '    exit 1',
            'fi',
            "",
            "ONNX_MODEL=$1",
            "TRT_MODEL=$2",
            "",
            "echo "Конвертируем $ONNX_MODEL в $TRT_MODEL"",
            "",
            "# Конвертация с использованием trtexec",
            "trtexec --onnx=$ONNX_MODEL \",
            "        --saveEngine=$TRT_MODEL \",
            "        --fp16 \",
            "        --minShapes=input:1x3x640x640 \",
            "        --optShapes=input:4x3x640x640 \",
            "        --maxShapes=input:8x3x640x640 \",
            "        --workspace=1024 \",
            "        --verbose",
            "",
            'if [ $? -eq 0 ]; then',
            '    echo "Конвертация завершена успешно!"',
            '    echo "TensorRT модель сохранена в: $TRT_MODEL"',
            'else',
            '    echo "Ошибка при конвертации!"',
            '    exit 1',
            'fi'
        ]

        script_content = "\n".join(script_lines)

        with open(output_path, 'w') as f:
            f.write(script_content)

        # Делаем скрипт исполняемым
        import os
        os.chmod(output_path, 0o755)

        print(f"Скрипт конвертации создан: {output_path}")

    @hydra.main(version_base=None, config_path="../../configs", config_name="deployment")
    def run(self, config: DictConfig):
        """
        Запуск конвертации с конфигурацией Hydra.

        Args:
            config: Конфигурация Hydra
        """
        print("Запуск конвертации ONNX модели в TensorRT...")

        # Проверяем наличие ONNX модели
        onnx_path = Path(config.deployment.onnx.output_path)
        if not onnx_path.exists():
            print(f"ONNX модель не найдена: {onnx_path}")
            print("Сначала запустите конвертацию в ONNX")
            return False

        # Создаем выходную директорию
        tensorrt_path = Path(config.deployment.tensorrt.output_path)
        tensorrt_path.parent.mkdir(parents=True, exist_ok=True)

        # Создаем скрипт конвертации
        script_path = tensorrt_path.parent / "convert_to_tensorrt.sh"
        self.create_conversion_script(script_path)

        # Конвертируем модель
        success = self.convert_onnx_to_tensorrt(
            onnx_path=onnx_path,
            output_path=tensorrt_path,
            precision=config.deployment.tensorrt.precision,
            max_batch_size=config.deployment.tensorrt.max_batch_size
        )

        if success:
            print("Конвертация в TensorRT завершена успешно!")
        else:
            print("Конвертация в TensorRT завершилась с ошибкой!")
            print(f"Вы можете попробовать использовать скрипт: {script_path}")

        return success


def main():
    """Точка входа для конвертации в TensorRT."""
    converter = TensorRTConverter()
    converter.run()


if __name__ == "__main__":
    main()
