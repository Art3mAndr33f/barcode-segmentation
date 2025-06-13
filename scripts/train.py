import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Настройка MLflow
    mlflow.set_experiment(cfg.experiment_name)
    
    with mlflow.start_run():
        # Логирование конфигурации
        mlflow.log_params(cfg.model)
        
        # Здесь будет код обучения модели
        print(f"Обучение модели: {cfg.model._target_}")
        print(f"Данные: {cfg.paths.data_dir}")
        
        # Пример логирования метрик
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("loss", 0.1)

if __name__ == "__main__":
    main()
EOF