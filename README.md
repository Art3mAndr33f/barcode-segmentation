# Barcode Detection MLOps Project

## 📋 Описание проекта

Этот проект представляет собой полноценный MLOps пайплайн для детекции и сегментации баркодов с использованием современных инструментов и лучших практик. Проект основан на архитектуре Detectron2 для instance segmentation и включает в себя все этапы от загрузки данных до развертывания модели в продакшен.

### 🎯 Основные возможности

- **Instance Segmentation**: Точная детекция и сегментация одномерных и двумерных баркодов
- **MLOps Pipeline**: Полный цикл разработки с использованием современных инструментов
- **Production Ready**: Готовые решения для развертывания в продакшен
- **Monitoring & Tracking**: Отслеживание экспериментов и мониторинг модели

### 🛠 Технологический стек

- **ML Framework**: PyTorch, Detectron2, PyTorch Lightning
- **MLOps Tools**: MLflow, DVC, Hydra
- **Code Quality**: Pre-commit, Black, Flake8, MyPy
- **Package Management**: Poetry
- **Production**: ONNX, TensorRT, MLflow Serving, Triton Inference Server
- **Containerization**: Docker

## 🚀 Быстрый старт

### Prerequisites

- Python 3.9+
- Git
- CUDA-compatible GPU (рекомендуется)
- Docker (для развертывания)

### Установка

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/Art3mAndr33f/barcode-segmentation.git
cd barcode-segmentation
```

2. **Установите Poetry:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. **Установите зависимости:**
```bash
poetry install
poetry shell
```

4. **Установите pre-commit hooks:**
```bash
pre-commit install
```

5. **Инициализируйте DVC:**
```bash
dvc init --no-scm
dvc remote add -d storage ./data/dvc-storage
```

## 📁 Структура проекта

```
barcode-segmentation/
├── .github/                     # GitHub Actions workflows
│   └── workflows/
│       ├── ci.yml              # CI/CD пайплайн
│       └── deploy.yml          # Развертывание
├── .dvc/                       # DVC конфигурация
├── configs/                    # Hydra конфигурации
│   ├── config.yaml            # Основной конфиг
│   ├── model/                 # Конфиги моделей
│   │   ├── detectron2.yaml   # Конфиг Detectron2
│   │   └── yolo.yaml         # Альтернативная модель
│   ├── data/                  # Конфиги данных
│   │   └── coco.yaml         # Формат COCO
│   └── train/                 # Конфиги обучения
│       └── default.yaml      # Параметры обучения
├── data/                      # Данные (версионируются DVC)
│   ├── raw/                  # Исходные данные
│   ├── processed/            # Обработанные данные
│   └── annotations/          # Аннотации
├── src/                      # Исходный код
│   ├── __init__.py
│   ├── data/                 # Модули работы с данными
│   │   ├── __init__.py
│   │   ├── dataset.py       # Датасеты
│   │   └── transforms.py    # Трансформации
│   ├── models/               # Модели
│   │   ├── __init__.py
│   │   ├── detectron_model.py
│   │   └── base_model.py
│   ├── training/            # Обучение
│   │   ├── __init__.py
│   │   ├── trainer.py      # Lightning модуль
│   │   └── callbacks.py    # Callbacks
│   ├── inference/          # Инференс
│   │   ├── __init__.py
│   │   ├── predictor.py   # Предсказания
│   │   └── api.py         # FastAPI сервер
│   └── utils/             # Утилиты
│       ├── __init__.py
│       ├── logging.py
│       └── metrics.py
├── tests/                # Тесты
│   ├── __init__.py
│   ├── test_data/
│   ├── test_models/
│   └─
```

## 🎯 Использование

### Setup

1. **Настройка окружения:**
```bash
# Активация виртуального окружения
poetry shell

# Проверка установки
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import detectron2; print('Detectron2 установлен успешно')"
```

2. **Загрузка данных:**
```bash
# Запуск скрипта загрузки данных
python scripts/download_data.py

# Или добавление собственных данных через DVC
dvc add data/raw/your_dataset
git add data/raw/your_dataset.dvc .gitignore
git commit -m "Add dataset"
```

### Train

Запуск обучения модели с различными конфигурациями:

```bash
# Базовое обучение
python -m barcode_detection.training.train

# Обучение с кастомными параметрами
python -m barcode_detection.training.train \
    training.max_epochs=50 \
    training.optimizer.lr=0.001 \
    model.roi_heads.batch_size_per_image=256

# Обучение с определенной конфигурацией
python -m barcode_detection.training.train \
    --config-name=config \
    --config-path=configs

# Запуск с Fire CLI
python commands.py train --max_epochs=100 --lr=0.001
```

#### Мониторинг обучения

1. **Запуск MLflow UI:**
```bash
mlflow ui --host 127.0.0.1 --port 8080
```

2. **Просмотр метрик в браузере:**
```
http://127.0.0.1:8080
```

### Production Preparation

#### 1. Экспорт в ONNX

```bash
# Экспорт модели в ONNX формат
python -m barcode_detection.export.onnx_export \
    --model_path models/best_model.pth \
    --output_path models/model.onnx \
    --input_shape 3 800 1333

# Валидация ONNX модели
python -m barcode_detection.export.onnx_export \
    --validate \
    --onnx_path models/model.onnx
```

#### 2. Конвертация в TensorRT

```bash
# Конвертация ONNX модели в TensorRT
bash convert_to_tensorrt.sh models/model.onnx models/model.trt

# Или через Python скрипт
python -m barcode_detection.export.tensorrt_export \
    --onnx_path models/model.onnx \
    --output_path models/model.trt \
    --precision fp16
```

### Infer

Запуск предсказаний на новых данных:

```bash
# Инференс на одном изображении
python -m barcode_detection.inference.infer \
    --image_path data/test/image.jpg \
    --model_path models/best_model.pth \
    --output_dir outputs/predictions

# Batch инференс
python -m barcode_detection.inference.infer \
    --input_dir data/test/ \
    --model_path models/best_model.pth \
    --output_dir outputs/predictions \
    --batch_size 4

# Инференс с ONNX моделью
python -m barcode_detection.inference.infer \
    --model_path models/model.onnx \
    --model_type onnx \
    --image_path data/test/image.jpg

# Использование REST API
python -m barcode_detection.inference.server
# Затем отправка POST запроса на http://localhost:8000/predict
```

#### Формат входных данных

Поддерживаемые форматы:
- **Изображения**: JPG, PNG, JPEG
- **Размеры**: Минимум 100x100, максимум 4000x4000 пикселей
- **Batch размер**: Настраивается в конфигурации

Пример структуры данных:
```
data/test/
├── image1.jpg
├── image2.png
└── image3.jpeg
```

## 🔧 Конфигурация

Проект использует Hydra для управления конфигурациями. Основные параметры находятся в папке `configs/`:

### Изменение параметров через командную строку

```bash
# Изменение learning rate
python -m barcode_detection.training.train training.optimizer.lr=0.01

# Изменение количества эпох
python -m barcode_detection.training.train training.max_epochs=200

# Использование GPU
python -m barcode_detection.training.train device.gpu_id=1

# Изменение размера батча
python -m barcode_detection.training.train data.dataloader.batch_size=4
```

### Создание кастомных конфигураций

Создайте новый файл конфигурации в соответствующей папке:

```yaml
# configs/training/custom_train.yaml
# @package _global_

training:
  max_epochs: 150
  optimizer:
    lr: 0.005
    weight_decay: 0.0005
```

Запуск с кастомной конфигурацией:
```bash
python -m barcode_detection.training.train --config-name=custom_train
```

## 📊 Мониторинг и Логирование

### MLflow Tracking

Все эксперименты автоматически логируются в MLflow:

- **Параметры**: Learning rate, batch size, архитектура модели
- **Метрики**: Loss, mAP, Precision, Recall
- **Артефакты**: Модели, графики, конфигурации
- **Система**: Git commit, версия кода, environment

### Просмотр результатов

1. Запустите MLflow UI:
```bash
mlflow ui
```

2. Откройте в браузере: http://127.0.0.1:5000

3. Сравните эксперименты и метрики

## 🔄 Data Version Control (DVC)

### Управление данными

```bash
# Добавление новых данных
dvc add data/raw/new_dataset
git add data/raw/new_dataset.dvc .gitignore
git commit -m "Add new dataset"

# Пуш данных в удаленное хранилище
dvc push

# Получение данных
dvc pull

# Переключение между версиями
git checkout data-v2
dvc checkout
```

### Пайплайны DVC

```bash
# Запуск полного пайплайна
dvc repro

# Просмотр DAG пайплайна
dvc dag

# Просмотр метрик
dvc metrics show
```

## 🧪 Тестирование

```bash
# Запуск всех тестов
pytest

# Запуск с покрытием
pytest --cov=barcode_detection --cov-report=html

# Запуск конкретного теста
pytest tests/test_models.py::TestDetectronModel::test_model_forward
```

## 🚀 Развертывание

### MLflow Serving

```bash
# Регистрация модели
python scripts/register_model.py

# Запуск MLflow serving
mlflow models serve -m "models:/barcode-detection/1" -p 5000

# Тестирование API
curl -X POST "http://127.0.0.1:5000/invocations" \
     -H "Content-Type: application/json" \
     -d @test_data.json
```

### Triton Inference Server

```bash
# Подготовка модели для Triton
python scripts/prepare_triton_model.py

# Запуск Triton сервера
docker run --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v${PWD}/triton_models:/models \
  nvcr.io/nvidia/tritonserver:23.04-py3 \
  tritonserver --model-repository=/models
```

### Docker

```bash
# Сборка образа
docker build -t barcode-detection:latest .

# Запуск контейнера
docker run -p 8000:8000 barcode-detection:latest
```

## 📈 Мониторинг производительности

### Метрики модели

- **mAP (mean Average Precision)**: Основная метрика качества
- **Precision/Recall**: По классам и общие
- **IoU (Intersection over Union)**: Качество детекции
- **Inference Time**: Скорость предсказаний

### Системные метрики

- **GPU Utilization**: Использование видеокарты
- **Memory Usage**: Потребление памяти
- **Throughput**: Количество обработанных изображений в секунду

## 🔧 Troubleshooting

### Частые проблемы

1. **CUDA Out of Memory**:
   ```bash
   # Уменьшите batch size
   python -m barcode_detection.training.train data.dataloader.batch_size=1
   ```

2. **DVC remote не настроен**:
   ```bash
   dvc remote add -d storage /path/to/storage
   ```

3. **MLflow server недоступен**:
   ```bash
   mlflow server --host 0.0.0.0 --port 5000
   ```

## 🤝 Контрибуция

1. Fork проекта
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📞 Контакты

- **Автор**: Андреев Артем
- **Email**: andreev.artem@phystech.edu
- **Проект**: [https://github.com/Art3mAndr33f/barcode-segmentation](https://github.com/Art3mAndr33f/barcode-segmentation)

## 🙏 Благодарности

- [Detectron2](https://github.com/facebookresearch/detectron2) - Основная архитектура для instance segmentation
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - Фреймворк для обучения
- [MLflow](https://github.com/mlflow/mlflow) - Отслеживание экспериментов
- [DVC](https://github.com/iterative/dvc) - Версионирование данных
- [Hydra](https://github.com/facebookresearch/hydra) - Управление конфигурациями
