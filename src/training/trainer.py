import os
import logging
from typing import Dict, Any
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader
import mlflow
import mlflow.pytorch
from omegaconf import DictConfig, OmegaConf
import torch

logger = logging.getLogger(__name__)

class MLflowHook(HookBase):
    """Hook для логирования в MLflow во время обучения."""
    
    def __init__(self):
        self.iteration = 0
    
    def after_step(self):
        # Логирование метрик после каждого шага
        if hasattr(self.trainer.storage, 'latest'):
            metrics = self.trainer.storage.latest()
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=self.iteration)
        self.iteration += 1

class BarcodeTrainer(DefaultTrainer):
    """Пользовательский trainer для баркодов с интеграцией MLflow."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Создание evaluator для оценки модели."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        """Добавление пользовательских hooks."""
        hooks = super().build_hooks()
        hooks.insert(-1, MLflowHook())  # Добавляем MLflow hook
        return hooks

def train_model(cfg: DictConfig) -> None:
    """
    Основная функция обучения модели.
    
    Args:
        cfg: Hydra конфигурация
    """
    from ..data.dataset import register_dataset, download_data_with_dvc
    from ..models.detectron_model import get_detectron2_cfg
    
    # Настройка MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    with mlflow.start_run():
        # Логирование параметров
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        # Загрузка данных через DVC
        if cfg.data.use_dvc:
            download_data_with_dvc(cfg.data.dvc_data_path)
        
        # Регистрация датасета
        register_dataset(cfg.data.train_dataset, cfg.data.train_data_dir)
        if cfg.data.test_datasets:
            for test_dataset, test_dir in zip(cfg.data.test_datasets, cfg.data.test_data_dirs):
                register_dataset(test_dataset, test_dir)
        
        # Создание конфигурации Detectron2
        detectron_cfg = get_detectron2_cfg(cfg.model, cfg.data, cfg.train)
        
        # Создание и обучение модели
        trainer = BarcodeTrainer(detectron_cfg)
        trainer.resume_or_load(resume=cfg.train.resume_training)
        
        logger.info("Starting training...")
        trainer.train()
        
        # Логирование артефактов
        model_path = os.path.join(detectron_cfg.OUTPUT_DIR, "model_final.pth")
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path, "model")
            logger.info(f"Model saved and logged to MLflow: {model_path}")
        
        # Оценка модели
        if cfg.data.test_datasets:
            evaluator = BarcodeTrainer.build_evaluator(
                detectron_cfg, 
                cfg.data.test_datasets[0],
                os.path.join(detectron_cfg.OUTPUT_DIR, "evaluation")
            )
            test_loader = build_detection_test_loader(detectron_cfg, cfg.data.test_datasets[0])
            results = trainer.test(detectron_cfg, trainer.model, evaluator)
            
            # Логирование результатов оценки
            for metric_name, metric_value in results.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"eval_{metric_name}", metric_value)
    
    logger.info("Training completed successfully")

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        train_model(cfg)
    
    main()
