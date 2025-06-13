import logging
import hydra
from omegaconf import DictConfig
from pathlib import Path
import sys

# Добавляем путь к src для импортов
sys.path.append(str(Path(__file__).parent))

from training.trainer import train_model
from inference.predictor import run_inference
from utils.export_model import export_to_onnx, export_to_tensorrt
from data.dataset import download_data_with_dvc

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Главная функция для запуска различных команд MLOps пайплайна.
    
    Команды запуска:
    python src/commands.py +command=train
    python src/commands.py +command=infer  
    python src/commands.py +command=export
    python src/commands.py +command=download_data
    """
    
    # Настройка логирования
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format
    )
    
    command = cfg.get("command", "train")
    
    logger.info(f"Executing command: {command}")
    
    if command == "train":
        train_model(cfg)
    elif command == "infer":
        run_inference(cfg)
    elif command == "export":
        export_models(cfg)
    elif command == "download_data":
        download_data_with_dvc(cfg.data.dvc_data_path)
    else:
        logger.error(f"Unknown command: {command}")
        raise ValueError(f"Unknown command: {command}")

def export_models(cfg: DictConfig) -> None:
    """Экспорт модели в различные форматы."""
    if cfg.model.export.onnx.enabled:
        export_to_onnx(cfg)
    
    if cfg.model.export.tensorrt.enabled:
        export_to_tensorrt(cfg)

if __name__ == "__main__":
    main()
