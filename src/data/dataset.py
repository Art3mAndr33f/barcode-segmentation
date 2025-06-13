import os
import json
import cv2
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import logging
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
import dvc.api

logger = logging.getLogger(__name__)

def get_barcode_dicts(data_dir: str) -> List[Dict[str, Any]]:
    """
    Функция для загрузки данных о баркодах из json файлов.

    Args:
        data_dir (str): Путь к директории с изображениями и json файлами.

    Returns:
        list: Список словарей, каждый из которых представляет изображение
              и содержит информацию о баркодах.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Data directory not found: {data_dir}")
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    dataset_dicts = []
    
    logger.info(f"Found {len(image_files)} images in {data_dir}")
    
    for idx, img_file in enumerate(image_files):
        json_file = os.path.join(data_dir, img_file.replace('.jpg', '.jpg.json'))
        try:
            with open(json_file) as f:
                img_anns = json.load(f)
        except FileNotFoundError:
            logger.warning(f"JSON file not found: {json_file}")
            continue

        record = {}
        filename = os.path.join(data_dir, img_file)
        
        # Get image dimensions
        img = cv2.imread(filename)
        if img is None:
            logger.warning(f"Could not load image: {filename}")
            continue
            
        real_height, real_width = img.shape[:2]
        width, height = img_anns.get('size', [real_width, real_height])
        
        # Handle size mismatch
        if real_width != width or real_height != height:
            logger.warning(f"Size mismatch for {filename}. Annotation: ({width}, {height}), Actual: ({real_width}, {real_height})")
            width, height = real_width, real_height

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        # Process annotations
        annos = img_anns.get('objects', [])
        objs = []
        for anno in annos:
            try:
                poly = np.array(anno['data']).astype(np.int32)
                px = poly[:, 0]
                py = poly[:, 1]
                bbox = [np.min(px), np.min(py), np.max(px), np.max(py)]
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                
                # Create segmentation mask
                segmentation = [poly.flatten().tolist()]
                
                obj = {
                    "bbox": [bbox[0], bbox[1], bbox_width, bbox_height],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": segmentation,
                    "category_id": 0,  # Barcode is category 0
                    "iscrowd": 0,
                }
                objs.append(obj)
            except Exception as e:
                logger.error(f"Error processing annotation in {img_file}: {e}")
                continue
                
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    logger.info(f"Successfully processed {len(dataset_dicts)} images")
    return dataset_dicts

def register_dataset(name: str, data_dir: str) -> None:
    """Регистрация датасета в Detectron2."""
    DatasetCatalog.register(name, lambda d=data_dir: get_barcode_dicts(d))
    MetadataCatalog.get(name).set(thing_classes=["barcode"])
    logger.info(f"Dataset '{name}' registered with data from {data_dir}")

def download_data_with_dvc(data_path: str) -> None:
    """Загрузка данных через DVC API."""
    try:
        with dvc.api.open(data_path, mode='r') as f:
            # DVC автоматически скачает данные при обращении
            pass
        logger.info(f"Data successfully accessed via DVC: {data_path}")
    except Exception as e:
        logger.error(f"Failed to download data via DVC: {e}")
        raise
