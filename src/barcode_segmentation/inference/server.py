#!/usr/bin/env python3
"""
FastAPI сервер для инференса модели сегментации штрих-кодов.

Предоставляет REST API для загрузки изображений и получения предсказаний модели.
Поддерживает асинхронную обработку и валидацию входных данных.

Endpoints:
- GET /health - проверка здоровья сервера
- POST /predict - предсказание для изображения
- GET /metrics - метрики сервера
- GET /docs - Swagger документация
"""

import asyncio
import io
import logging
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

from barcode_segmentation.inference.predictor import BarcodePredictor
from barcode_segmentation.utils.visualizer import Visualizer

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Модель запроса для предсказания."""
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    return_visualization: bool = Field(default=False)
    output_format: str = Field(default="json", regex="^(json|coco)$")


class BoundingBox(BaseModel):
    """Модель для bounding box."""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate") 
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")
    confidence: float = Field(..., ge=0.0, le=1.0)


class SegmentationMask(BaseModel):
    """Модель для маски сегментации."""
    polygon: List[List[float]] = Field(..., description="Polygon coordinates")
    area: float = Field(..., ge=0.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    """Модель результата предсказания."""
    bounding_boxes: List[BoundingBox] = Field(default_factory=list)
    segmentation_masks: List[SegmentationMask] = Field(default_factory=list)
    processing_time: float = Field(..., description="Processing time in seconds")
    image_shape: tuple = Field(..., description="Original image dimensions (H, W, C)")
    num_detections: int = Field(..., description="Number of detected barcodes")


class HealthResponse(BaseModel):
    """Модель ответа health check."""
    status: str
    timestamp: float
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    """Модель ответа с метриками."""
    total_requests: int
    total_predictions: int
    average_processing_time: float
    uptime_seconds: float


class InferenceServer:
    """
    Сервер для инференса модели сегментации штрих-кодов.
    
    Поддерживает:
    - Асинхронную обработку запросов
    - Валидацию входных данных
    - Метрики производительности  
    - Health checks
    - CORS для веб-интеграции
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        model_path: Optional[str] = None,
        config_path: str = "configs/inference.yaml"
    ):
        """
        Инициализация сервера.
        
        Args:
            host: Хост для сервера
            port: Порт для сервера
            model_path: Путь к модели
            config_path: Путь к конфигурации
        """
        self.host = host
        self.port = port
        self.model_path = model_path
        self.config_path = config_path
        
        # Метрики
        self.start_time = time.time()
        self.total_requests = 0
        self.total_predictions = 0
        self.processing_times = []
        
        # Инициализация FastAPI
        self.app = FastAPI(
            title="Barcode Segmentation API",
            description="REST API для сегментации штрих-кодов",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Настройка CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Инициализация модели
        self.predictor = None
        self.visualizer = None
        self._setup_routes()

    async def _load_model(self) -> None:
        """Асинхронная загрузка модели."""
        try:
            logger.info("Загрузка модели...")
            self.predictor = BarcodePredictor()
            self.predictor.load_model(self.model_path, self.config_path)
            self.visualizer = Visualizer()
            logger.info("✅ Модель загружена успешно")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            raise

    def _setup_routes(self) -> None:
        """Настройка маршрутов API."""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Событие при запуске сервера."""
            await self._load_model()
            logger.info(f"🚀 Сервер запущен на {self.host}:{self.port}")

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Проверка здоровья сервера."""
            return HealthResponse(
                status="healthy" if self.predictor else "loading",
                timestamp=time.time(),
                model_loaded=self.predictor is not None,
                version="1.0.0"
            )

        @self.app.get("/metrics", response_model=MetricsResponse)
        async def get_metrics():
            """Получение метрик сервера."""
            avg_time = (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0.0
            )
            
            return MetricsResponse(
                total_requests=self.total_requests,
                total_predictions=self.total_predictions,
                average_processing_time=avg_time,
                uptime_seconds=time.time() - self.start_time
            )

        @self.app.post("/predict", response_model=PredictionResult)
        async def predict(
            file: UploadFile = File(...),
            confidence_threshold: float = 0.5,
            return_visualization: bool = False
        ):
            """
            Предсказание для загруженного изображения.
            
            Args:
                file: Загружаемое изображение
                confidence_threshold: Порог уверенности
                return_visualization: Возвращать ли визуализацию
                
            Returns:
                Результат предсказания с детекциями и масками
            """
            start_time = time.time()
            self.total_requests += 1
            
            try:
                # Проверка модели
                if not self.predictor:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Модель не загружена"
                    )
                
                # Валидация файла
                if not file.content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Файл должен быть изображением"
                    )
                
                # Чтение изображения
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(image)
                
                # Конвертация в BGR для OpenCV
                if len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # Предсказание
                predictions = await self._predict_async(
                    image_array, confidence_threshold
                )
                
                # Обработка результатов
                bboxes = []
                masks = []
                
                for pred in predictions:
                    # Bounding boxes
                    if "bbox" in pred:
                        bbox = pred["bbox"]
                        bboxes.append(BoundingBox(
                            x1=float(bbox[0]),
                            y1=float(bbox[1]),
                            x2=float(bbox[2]),
                            y2=float(bbox[3]),
                            confidence=float(pred.get("confidence", 0.0))
                        ))
                    
                    # Segmentation masks
                    if "mask" in pred:
                        mask_data = pred["mask"]
                        masks.append(SegmentationMask(
                            polygon=mask_data.get("polygon", []),
                            area=float(mask_data.get("area", 0.0)),
                            confidence=float(pred.get("confidence", 0.0))
                        ))
                
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.total_predictions += 1
                
                return PredictionResult(
                    bounding_boxes=bboxes,
                    segmentation_masks=masks,
                    processing_time=processing_time,
                    image_shape=image_array.shape,
                    num_detections=len(predictions)
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Ошибка при предсказании: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Ошибка обработки: {str(e)}"
                )

        @self.app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            """Глобальный обработчик исключений."""
            logger.error(f"Необработанная ошибка: {exc}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Внутренняя ошибка сервера"}
            )

    async def _predict_async(
        self, image: np.ndarray, confidence_threshold: float
    ) -> List[Dict]:
        """
        Асинхронное выполнение предсказания.
        
        Args:
            image: Входное изображение
            confidence_threshold: Порог уверенности
            
        Returns:
            Список предсказаний
        """
        loop = asyncio.get_event_loop()
        
        # Выполняем синхронную функцию в executor
        predictions = await loop.run_in_executor(
            None, 
            self.predictor.predict,
            image,
            confidence_threshold
        )
        
        return predictions

    def run(self) -> None:
        """Запуск сервера."""
        logger.info(f"🌐 Запуск inference сервера на {self.host}:{self.port}")
        
        try:
            uvicorn.run(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=True
            )
        except KeyboardInterrupt:
            logger.info("🛑 Сервер остановлен пользователем")
        except Exception as e:
            logger.error(f"❌ Ошибка запуска сервера: {e}")
            raise


def main() -> None:
    """Главная функция для запуска сервера."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model-path", help="Path to model")
    parser.add_argument("--config-path", default="configs/inference.yaml")
    
    args = parser.parse_args()
    
    server = InferenceServer(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        config_path=args.config_path
    )
    
    server.run()


if __name__ == "__main__":
    main()