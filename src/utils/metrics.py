import numpy as np
import cv2
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

def calculate_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Вычисляет IoU между двумя масками.

    Args:
        mask1 (np.ndarray): Первая маска (логический массив).
        mask2 (np.ndarray): Вторая маска (логический массив).

    Returns:
        float: IoU.
    """
    if mask1.shape != mask2.shape:
        logger.warning(f"Mask shapes don't match: {mask1.shape} vs {mask2.shape}")
        return 0.0
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0  # Prevent division by zero
    
    iou = intersection / union
    return float(iou)

def calculate_modified_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """
    Вычисляет модифицированный IoU: f(P, T) = (S1 + 100 * S2) / ST,
    где P - предсказанная маска, T - размеченная маска,
    S1 - площадь области P \ T, S2 - площадь области T \ P, ST - площадь области T.

    Args:
        predicted_mask (np.ndarray): Предсказанная маска (логический массив).
        ground_truth_mask (np.ndarray): Размеченная маска (логический массив).

    Returns:
        float: Модифицированный IoU.
    """
    if predicted_mask.shape != ground_truth_mask.shape:
        logger.warning(f"Mask shapes don't match: {predicted_mask.shape} vs {ground_truth_mask.shape}")
        return 0.0
    
    s1 = np.logical_and(predicted_mask, np.logical_not(ground_truth_mask)).sum()
    s2 = np.logical_and(ground_truth_mask, np.logical_not(predicted_mask)).sum()
    st = ground_truth_mask.sum()

    if st == 0:
        return 0.0  # Prevent division by zero

    modified_iou = (s1 + 100 * s2) / st
    return float(modified_iou)

def calculate_batch_metrics(predictions: List[np.ndarray], 
                          ground_truths: List[np.ndarray]) -> Dict[str, float]:
    """
    Вычисляет метрики для батча предсказаний.
    
    Args:
        predictions: Список предсказанных масок
        ground_truths: Список истинных масок
        
    Returns:
        Dict с средними метриками
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Number of predictions and ground truths must match")
    
    iou_scores = []
    modified_iou_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        iou = calculate_iou(pred, gt)
        modified_iou = calculate_modified_iou(pred, gt)
        
        iou_scores.append(iou)
        modified_iou_scores.append(modified_iou)
    
    metrics = {
        "mean_iou": np.mean(iou_scores),
        "std_iou": np.std(iou_scores),
        "mean_modified_iou": np.mean(modified_iou_scores),
        "std_modified_iou": np.std(modified_iou_scores),
        "num_samples": len(predictions)
    }
    
    logger.info(f"Batch metrics calculated for {len(predictions)} samples")
    return metrics

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Преобразует полигон в маску.
    
    Args:
        polygon: Список точек полигона [(x1, y1), (x2, y2), ...]
        image_shape: Размер изображения (height, width)
        
    Returns:
        Бинарная маска
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)

def mask_to_polygon(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Преобразует маску в полигоны.
    
    Args:
        mask: Бинарная маска
        
    Returns:
        Список полигонов
    """
    # Находим контуры
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        # Упрощаем контур
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Преобразуем в список точек
        polygon = [(int(point[0][0]), int(point[0][1])) for point in approx]
        if len(polygon) >= 3:  # Минимум 3 точки для полигона
            polygons.append(polygon)
    
    return polygons
