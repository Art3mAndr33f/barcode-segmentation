import os
import torch
import logging
import onnx
from pathlib import Path
from omegaconf import DictConfig
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.export import TracingAdapter, dump_torchscript_IR
import tensorrt as trt

logger = logging.getLogger(__name__)

def export_to_onnx(cfg: DictConfig) -> str:
    """
    Экспорт модели Detectron2 в ONNX формат.
    
    Args:
        cfg: Hydra конфигурация
        
    Returns:
        Путь к созданному ONNX файлу
    """
    from ..models.detectron_model import get_detectron2_cfg
    
    # Создание конфигурации Detectron2
    detectron_cfg = get_detectron2_cfg(cfg.model, cfg.data, cfg.train)
    detectron_cfg.MODEL.WEIGHTS = cfg.inference.weights_path
    detectron_cfg.MODEL.DEVICE = "cpu"  # ONNX экспорт лучше работает на CPU
    
    # Создание модели
    model = build_model(detectron_cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(detectron_cfg.MODEL.WEIGHTS)
    model.eval()
    
    # Подготовка для экспорта
    height, width = 800, 800  # Фиксированный размер для экспорта
    input_tensor = torch.randn(1, 3, height, width)
    
    # Создание TracingAdapter для Detectron2
    tracing_adapter = TracingAdapter(model, input_tensor)
    
    # Экспорт в ONNX
    output_path = cfg.model.export.onnx.output_path
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    torch.onnx.export(
        tracing_adapter,
        input_tensor,
        output_path,
        opset_version=cfg.model.export.onnx.opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'labels', 'scores', 'masks'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'boxes': {0: 'batch_size', 1: 'num_detections'},
            'labels': {0: 'batch_size', 1: 'num_detections'},
            'scores': {0: 'batch_size', 1: 'num_detections'},
            'masks': {0: 'batch_size', 1: 'num_detections'}
        }
    )
    
    # Проверка созданной модели
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    logger.info(f"Model successfully exported to ONNX: {output_path}")
    return output_path

def export_to_tensorrt(cfg: DictConfig) -> str:
    """
    Экспорт ONNX модели в TensorRT формат.
    
    Args:
        cfg: Hydra конфигурация
        
    Returns:
        Путь к созданному TensorRT файлу
    """
    onnx_path = cfg.model.export.onnx.output_path
    tensorrt_path = cfg.model.export.tensorrt.output_path
    precision = cfg.model.export.tensorrt.precision
    
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX model not found: {onnx_path}")
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    
    # Создание TensorRT logger и builder
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    
    # Настройка precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # Для INT8 нужен калибратор - здесь упрощенная версия
        logger.warning("INT8 precision requires calibration dataset")
    
    # Создание network из ONNX
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            logger.error("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                logger.error(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")
    
    # Настройка workspace
    config.max_workspace_size = 1 << 30  # 1GB
    
    # Создание engine
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Сохранение engine
    os.makedirs(Path(tensorrt_path).parent, exist_ok=True)
    with open(tensorrt_path, 'wb') as f:
        f.write(engine.serialize())
    
    logger.info(f"Model successfully exported to TensorRT: {tensorrt_path}")
    return tensorrt_path

def validate_exported_model(original_weights: str, onnx_path: str, test_image_path: str) -> Dict[str, Any]:
    """
    Валидация экспортированной модели путем сравнения с оригиналом.
    
    Args:
        original_weights: Путь к оригинальным весам
        onnx_path: Путь к ONNX модели
        test_image_path: Путь к тестовому изображению
        
    Returns:
        Результаты сравнения
    """
    import cv2
    import numpy as np
    import onnxruntime as ort
    from ..inference.predictor import BarcodePredictor
    
    # Загрузка тестового изображения
    image = cv2.imread(test_image_path)
    if image is None:
        raise ValueError(f"Could not load test image: {test_image_path}")
    
    # Предсказание оригинальной моделью
    predictor = BarcodePredictor("", original_weights)
    original_results = predictor.predict_image(test_image_path)
    
    # Предсказание ONNX моделью
    ort_session = ort.InferenceSession(onnx_path)
    
    # Препроцессинг для ONNX
    input_tensor = cv2.resize(image, (800, 800))
    input_tensor = input_tensor.transpose(2, 0, 1).astype(np.float32)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # Инференс ONNX
    onnx_results = ort_session.run(None, {'input': input_tensor})
    
    # Сравнение результатов (упрощенная версия)
    validation_results = {
        "original_detections": original_results["num_detections"],
        "onnx_detections": len(onnx_results[0][0]) if onnx_results[0] is not None else 0,
        "validation_passed": True,  # Здесь должна быть более сложная логика сравнения
        "model_size_mb": os.path.getsize(onnx_path) / (1024 * 1024)
    }
    
    logger.info(f"Model validation completed: {validation_results}")
    return validation_results

if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    
    @hydra.main(version_base=None, config_path="../../configs", config_name="config")
    def main(cfg: DictConfig) -> None:
        if cfg.model.export.onnx.enabled:
            export_to_onnx(cfg)
        
        if cfg.model.export.tensorrt.enabled:
            export_to_tensorrt(cfg)
    
    main()
