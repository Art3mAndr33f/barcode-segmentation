import os
import cv2
import json
import numpy as np
import torch
import logging
from typing import List, Dict, Any, Tuple
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class BarcodePredictor:
    """Класс для инференса модели детекции баркодов."""
    
    def __init__(self, config_path: str, weights_path: str, device: str = "cuda"):
        """
        Инициализация предиктора.
        
        Args:
            config_path: Путь к конфигурации модели  
            weights_path: Путь к весам модели
            device: Устройство для инференса (cuda/cpu)
        """
        self.cfg = self._setup_config(config_path, weights_path, device)
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get("barcode_train")  # Предполагаем, что metadata уже зарегистрирована
        logger.info(f"BarcodePredictor initialized with weights: {weights_path}")
    
    def _setup_config(self, config_path: str, weights_path: str, device: str):
        """Настройка конфигурации для инференса."""
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.DEVICE = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        return cfg
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """
        Предсказание для одного изображения.
        
        Args:
            image_path: Путь к изображению
            
        Returns:
            Dict с результатами предсказания
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Загрузка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Получение предсказаний
        outputs = self.predictor(image)
        
        # Извлечение результатов
        instances = outputs["instances"].to("cpu")
        
        results = {
            "image_path": image_path,
            "image_size": image.shape[:2],
            "num_detections": len(instances),
            "boxes": instances.pred_boxes.tensor.numpy().tolist() if len(instances) > 0 else [],
            "scores": instances.scores.numpy().tolist() if len(instances) > 0 else [],
            "masks": instances.pred_masks.numpy().tolist() if len(instances) > 0 and instances.has("pred_masks") else []
        }
        
        logger.info(f"Predicted {results['num_detections']} barcodes in {image_path}")
        return results
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Предсказание для батча изображений.
        
        Args:
            image_paths: Список путей к изображениям
            
        Returns:
            Список результатов предсказаний
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({"image_path": image_path, "error": str(e)})
        
        return results
    
    def visualize_predictions(self, image_path: str, output_path: str = None) -> str:
        """
        Визуализация предсказаний на изображении.
        
        Args:
            image_path: Путь к исходному изображению
            output_path: Путь для сохранения результата
            
        Returns:
            Путь к сохраненному изображению с визуализацией
        """
        image = cv2.imread(image_path)
        outputs = self.predictor(image)
        
        # Создание визуализации
        v = Visualizer(image[:, :, ::-1], self.metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Сохранение результата
        if output_path is None:
            output_path = image_path.replace('.jpg', '_predicted.jpg')
        
        cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
        logger.info(f"Visualization saved to {output_path}")
        return output_path

def run_inference(cfg: DictConfig) -> None:
    """
    Функция для запуска инференса из командной строки.
    
    Args:
        cfg: Hydra конфигурация
    """
    from ..utils.metrics import calculate_iou, calculate_modified_iou
    
    # Создание предиктора
    predictor = BarcodePredictor(
        config_path=cfg.model.config_file,
        weights_path=cfg.inference.weights_path,
        device=cfg.inference.device
    )
    
    # Определение входных данных
    if cfg.inference.input_type == "single_image":
        results = [predictor.predict_image(cfg.inference.input_path)]
    elif cfg.inference.input_type == "directory":
        image_paths = [
            os.path.join(cfg.inference.input_path, f) 
            for f in os.listdir(cfg.inference.input_path) 
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ]
        results = predictor.predict_batch(image_paths)
    else:
        raise ValueError(f"Unsupported input type: {cfg.inference.input_type}")
    
    # Сохранение результатов
    output_file = os.path.join(cfg.inference.output_dir, "predictions.json")
    os.makedirs(cfg.inference.output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Inference results saved to {output_file}")
    
    # Создание визуализаций, если запрошено
    if cfg.inference.save_visualizations:
        for result in results:
            if "error" not in result:
                try:
                    vis_path = predictor.visualize_predictions(
                        result["image_path"],
                        os.path.join(cfg.inference.output_dir, f"vis_{Path(result['image_path']).name}")
                    )
                except Exception as e:
                    logger.error(f"Error creating visualization for {result['image_path']}: {e}")

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(version_base=None, config_path="../../configs", config_name="config")  
    def main(cfg: DictConfig) -> None:
        run_inference(cfg)
    
    main()
