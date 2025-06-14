#!/usr/bin/env python3
"""
Обертка над Detectron2 для задачи сегментации штрих-кодов.

Предоставляет интерфейс для использования Detectron2 в проекте, 
включая конфигурацию, загрузку предобученных моделей и инференс.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Detectron2Wrapper:
    """
    Обертка над Detectron2 для задачи сегментации штрих-кодов.
    
    Инкапсулирует работу с Detectron2, обеспечивая:
    - Конфигурацию модели
    - Загрузку предобученных весов
    - Преобразование данных между PyTorch и Detectron2
    - Предсказание и обучение
    
    Attributes:
        model: Модель Detectron2
        cfg: Конфигурация Detectron2
    """
    
    def __init__(self, config: DictConfig):
        """
        Инициализация обертки Detectron2.
        
        Args:
            config: Конфигурация модели
        """
        self.config = config
        self.model = None
        self.cfg = None
        
        # Инициализируем модель
        self._initialize_model()
        
        logger.info(f"Инициализирована обертка Detectron2 для {config.model_name}")
    
    def _initialize_model(self) -> None:
        """Инициализация модели Detectron2."""
        try:
            # Импортируем здесь, так как Detectron2 может быть недоступен
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.modeling import build_model
            
            # Создаем конфигурацию Detectron2
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(self.config.model_name))
            
            # Настраиваем модель
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config.model_name)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.config.num_classes
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.score_thresh_test
            cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.config.nms_thresh
            
            # Для обучения
            if hasattr(self.config, "device") and self.config.device:
                cfg.MODEL.DEVICE = self.config.device
            
            if hasattr(self.config, "batch_size") and self.config.batch_size:
                cfg.SOLVER.IMS_PER_BATCH = self.config.batch_size
            
            if hasattr(self.config, "base_lr") and self.config.base_lr:
                cfg.SOLVER.BASE_LR = self.config.base_lr
            
            if hasattr(self.config, "max_iter") and self.config.max_iter:
                cfg.SOLVER.MAX_ITER = self.config.max_iter
            
            # Сохраняем конфигурацию
            self.cfg = cfg
            
            # Создаем модель
            self.model = build_model(cfg)
            self.model.eval()
            
            logger.info(f"✓ Detectron2 модель инициализирована: {self.config.model_name}")
            
        except ImportError:
            logger.error("❌ Не удалось импортировать Detectron2.")
            logger.error("Пожалуйста, установите его: pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html")
            raise
    
    def forward(self, images: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Forward pass модели.
        
        Args:
            images: Список изображений (tensor)
            
        Returns:
            Список предсказаний
        """
        return self.forward_inference(images)
    
    def forward_train(
        self, images: List[torch.Tensor], targets: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass в режиме обучения.
        
        Args:
            images: Список изображений
            targets: Список целей
            
        Returns:
            Словарь с лоссами
        """
        # Конвертируем данные в формат Detectron2
        detectron2_inputs = self._prepare_detectron2_inputs(images, targets)
        
        # Запускаем forward pass с подсчетом градиентов
        self.model.train()
        with torch.enable_grad():
            loss_dict = self.model(detectron2_inputs)
        
        # Возвращаем лоссы
        return {k: v.mean() for k, v in loss_dict.items()}
    
    def forward_inference(
        self, images: List[torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """
        Forward pass в режиме инференса.
        
        Args:
            images: Список изображений
            
        Returns:
            Список предсказаний
        """
        # Конвертируем данные в формат Detectron2
        detectron2_inputs = self._prepare_detectron2_inputs(images)
        
        # Запускаем inference без подсчета градиентов
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(detectron2_inputs)
        
        # Преобразуем выходы Detectron2 в понятный формат
        predictions = self._process_detectron2_outputs(outputs)
        
        return predictions
    
    def _prepare_detectron2_inputs(
        self, 
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Подготовка входных данных для Detectron2.
        
        Args:
            images: Список изображений
            targets: Список целей (опционально)
            
        Returns:
            Список входных данных для Detectron2
        """
        from detectron2.structures import Boxes, Instances, BitMasks
        
        detectron2_inputs = []
        
        for i, image in enumerate(images):
            # Преобразуем формат и размерность
            if image.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:
                image_np = image.cpu().numpy()
            
            # Создаем входной словарь
            input_dict = {
                "image": torch.as_tensor(image_np, device=self.model.device).permute(2, 0, 1),
                "height": image_np.shape[0],
                "width": image_np.shape[1]
            }
            
            # Если есть цели, добавляем их
            if targets is not None and i < len(targets):
                target = targets[i]
                
                # Создаем instances для хранения аннотаций
                instances = Instances((image_np.shape[0], image_np.shape[1]))
                
                # Добавляем боксы
                if "boxes" in target:
                    instances.gt_boxes = Boxes(target["boxes"])
                
                # Добавляем маски
                if "masks" in target:
                    instances.gt_masks = BitMasks(target["masks"])
                
                # Добавляем классы
                if "classes" in target:
                    instances.gt_classes = target["classes"]
                
                input_dict["instances"] = instances
            
            detectron2_inputs.append(input_dict)
        
        return detectron2_inputs
    
    def _process_detectron2_outputs(
        self, outputs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Обработка выходных данных Detectron2.
        
        Args:
            outputs: Выходные данные Detectron2
            
        Returns:
            Список предсказаний в понятном формате
        """
        results = []
        
        for output in outputs:
            # Получаем instances
            instances = output["instances"]
            
            # Получаем данные с устройства
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            
            # Получаем маски, если есть
            if instances.has("pred_masks"):
                masks = instances.pred_masks.cpu().numpy()
            else:
                masks = None
            
            # Создаем результаты для каждого предсказания
            result = []
            for i in range(len(scores)):
                pred = {
                    "bbox": boxes[i].tolist(),
                    "confidence": float(scores[i]),
                    "class_id": int(classes[i]),
                }
                
                # Добавляем маску, если есть
                if masks is not None:
                    mask_data = {
                        "binary_mask": masks[i].astype(bool),
                        "polygon": self._mask_to_polygon(masks[i]),
                        "area": float(np.sum(masks[i]))
                    }
                    pred["mask"] = mask_data
                
                result.append(pred)
            
            results.append(result)
        
        return results
    
    def _mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        """
        Преобразование маски в полигон.
        
        Args:
            mask: Бинарная маска
            
        Returns:
            Полигон в формате [[x1, y1], [x2, y2], ...]
        """
        # Преобразуем маску в 8-битное изображение
        mask_8bit = (mask * 255).astype(np.uint8)
        
        # Находим контуры
        contours, _ = cv2.findContours(
            mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Берем самый большой контур
        if not contours:
            return []
            
        contour = max(contours, key=cv2.contourArea)
        
        # Преобразуем в список точек
        return [point[0].tolist() for point in contour]
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Получение state_dict модели.
        
        Returns:
            State dict модели
        """
        return self.model.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Загрузка state_dict модели.
        
        Args:
            state_dict: State dict для загрузки
        """
        self.model.load_state_dict(state_dict)
    
    def save_model(self, path: str) -> None:
        """
        Сохранение модели.
        
        Args:
            path: Путь для сохранения
        """
        # Создаем директорию, если нужно
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем модель
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }, path)
        
        logger.info(f"Модель сохранена: {path}")
    
    def load_model(self, path: str) -> None:
        """
        Загрузка модели.
        
        Args:
            path: Путь к модели
        """
        # Загружаем модель
        checkpoint = torch.load(path, map_location=self.model.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Модель загружена: {path}")


def main() -> None:
    """Основная функция для тестирования модуля."""
    # Создаем тестовую конфигурацию
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        "model_name": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "num_classes": 1,
        "score_thresh_test": 0.5,
        "nms_thresh": 0.5
    })
    
    # Создаем модель
    model = Detectron2Wrapper(config)
    print(f"Модель создана: {model}")


if __name__ == "__main__":
    main()