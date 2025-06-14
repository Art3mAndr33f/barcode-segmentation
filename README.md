# Barcode Segmentation MLOps Project

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![Detectron2](https://img.shields.io/badge/Detectron2-0.6+-006600.svg)](https://github.com/facebookresearch/detectron2)

Полнофункциональный MLOps проект для автоматической сегментации штрих-кодов с использованием современного стека технологий машинного обучения.

## 🎯 Описание проекта

Этот проект реализует полный pipeline машинного обучения для задачи сегментации штрих-кодов:

- **Сегментация штрих-кодов** на изображениях различного качества и сложности
- **Instance segmentation** с использованием Mask R-CNN архитектуры  
- **Производственный деплоймент** с REST API и контейнеризацией
- **Отслеживание экспериментов** через MLflow
- **Версионирование данных** с DVC
- **Автоматизированный CI/CD** с проверкой качества кода

### Ключевые особенности

✅ **Модульная архитектура** с PyTorch Lightning  
✅ **Современные практики MLOps** (Hydra, MLflow, DVC)  
✅ **Качество кода** (pre-commit, black, isort, flake8, mypy)  
✅ **Производственный деплоймент** (FastAPI, Docker, ONNX, TensorRT)  
✅ **Подробная документация** и примеры использования  
✅ **Мониторинг и метрики** в реальном времени  

## 🛠 Технологический стек

### Машинное обучение
- **Framework**: PyTorch 2.0+ + PyTorch Lightning
- **Модель**: Detectron2 (Mask R-CNN + ResNet-50 backbone)
- **Конфигурация**: Hydra для управления настройками
- **Эксперименты**: MLflow для tracking и логирования

### MLOps инфраструктура  
- **Управление зависимостями**: Poetry
- **Версионирование данных**: DVC
- **Качество кода**: pre-commit hooks (black, isort, flake8, mypy)
- **Контейнеризация**: Docker + Docker Compose

### Продакшен деплоймент
- **API**: FastAPI с автоматической документацией
- **Модели**: Экспорт в ONNX и TensorRT для оптимизации
- **Мониторинг**: Prometheus + Grafana метрики
- **Оркестрация**: Kubernetes (опционально)

## 📁 Структура проекта

```
barcode_segmentation/
├── 📂 src/barcode_segmentation/          # Исходный код
│   ├── 📂 data/                          # Модули работы с данными
│   │   ├── dataset.py                    # Dataset классы
│   │   ├── dataloader.py                 # DataModule для Lightning
│   │   └── preprocessing.py              # Предобработка данных
│   ├── 📂 models/                        # Модели и архитектуры
│   │   ├── detectron2_wrapper.py         # Обертка Detectron2
│   │   ├── lightning_module.py           # Lightning модуль
│   │   └── utils.py                      # Утилиты для моделей
│   ├── 📂 training/                      # Компоненты обучения
│   │   ├── trainer.py                    # Trainer класс
│   │   └── evaluator.py                  # Оценка моделей
│   ├── 📂 inference/                     # Инференс
│   │   ├── predictor.py                  # Предсказания
│   │   └── postprocessing.py             # Постобработка
│   ├── 📂 deployment/                    # Деплоймент
│   │   ├── inference_server.py           # FastAPI сервер
│   │   ├── onnx_converter.py             # ONNX экспорт
│   │   └── tensorrt_converter.py         # TensorRT оптимизация
│   └── 📂 utils/                         # Утилиты
│       ├── metrics.py                    # Метрики
│       └── visualizer.py                 # Визуализация
├── 📂 configs/                           # Hydra конфигурации
│   ├── config.yaml                       # Основная конфигурация
│   ├── train.yaml                        # Параметры обучения
│   ├── inference.yaml                    # Параметры инференса
│   └── 📂 model/                         # Конфигурации моделей
├── 📂 data/                              # Данные
│   ├── 📂 raw/                           # Исходные данные
│   ├── 📂 processed/                     # Обработанные данные
│   └── 📂 interim/                       # Промежуточные данные
├── 📂 tests/                             # Тесты
├── 📂 scripts/                           # Скрипты установки
├── 📂 notebooks/                         # Jupyter notebooks
├── commands.py                           # CLI интерфейс
├── pyproject.toml                        # Poetry конфигурация
├── Dockerfile                            # Docker образ
└── README.md                             # Документация
```

## 🚀 Быстрый старт

### 1. Клонирование и установка

```bash
# Клонируем репозиторий
git clone https://github.com/Art3mAndr33f/barcode_segmentation.git
cd barcode_segmentation

# Установка зависимостей через Poetry
poetry install

# Активация виртуального окружения
poetry shell

# Установка Detectron2 (для Windows используйте scripts/install_detectron2.py)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# Установка пакета в editable режиме
pip install -e .
```

### 2. Настройка окружения

```bash
# Инициализация всех компонентов
python commands.py setup_environment

# Настройка pre-commit hooks
pre-commit install

# Проверка здоровья системы
python commands.py health_check
```

### 3. Подготовка данных

```bash
# Поместите ваши данные в data/raw/:
# - .jpg файлы (изображения)
# - .jpg.json файлы (аннотации в COCO формате)

# Предобработка данных
python commands.py preprocess --config_path configs/data/preprocessing.yaml

# Настройка DVC (опционально)
python commands.py setup_dvc --remote_url s3://your-bucket/data
```

### 4. Обучение модели

```bash
# Запуск обучения с default конфигурацией
python commands.py train

# Обучение с кастомными параметрами
python commands.py train --config_path configs/train.yaml

# Обучение с переопределением параметров через CLI
python commands.py train training.max_epochs=20 model.learning_rate=0.001
```

### 5. Инференс и деплоймент

```bash
# Локальный инференс
python commands.py infer --config_path configs/inference.yaml

# Запуск inference сервера
python commands.py serve --host 0.0.0.0 --port 8000

# Экспорт модели в ONNX
python commands.py convert_to_onnx

# Экспорт в TensorRT (требует NVIDIA GPU)
python commands.py convert_to_tensorrt
```

## 📊 Мониторинг экспериментов

### MLflow

```bash
# Запуск MLflow UI
mlflow server --host 127.0.0.1 --port 8080

# Веб-интерфейс доступен по адресу:
# http://127.0.0.1:8080
```

### Интеграция с Weights & Biases (опционально)

```bash
# Установка wandb
poetry install --extras wandb

# Настройка в конфигурации
# logging.use_wandb: true
```

## 🔧 Основные команды

| Команда | Описание |
|---------|----------|
| `python commands.py preprocess` | Предобработка и валидация данных |
| `python commands.py train` | Обучение модели с логированием в MLflow |
| `python commands.py infer` | Инференс на новых данных |
| `python commands.py serve` | Запуск inference сервера |
| `python commands.py convert_to_onnx` | Конвертация модели в ONNX |
| `python commands.py convert_to_tensorrt` | Оптимизация для TensorRT |
| `python commands.py setup_dvc` | Настройка DVC |
| `python commands.py health_check` | Проверка здоровья системы |

## 🐳 Docker деплоймент

### Сборка и запуск контейнера

```bash
# Сборка образа
docker build -t barcode-segmentation .

# Запуск inference сервера
docker run -p 8000:8000 barcode-segmentation

# Запуск с подключением локальных данных
docker run -p 8000:8000 -v $(pwd)/data:/app/data barcode-segmentation
```

### Docker Compose

```bash
# Запуск полного стека (API + MLflow + база данных)
docker-compose up -d

# Остановка сервисов
docker-compose down
```

## 📈 Метрики и оценка

Проект отслеживает следующие метрики:

- **IoU** (Intersection over Union) - стандартная метрика сегментации
- **Modified IoU** - кастомная метрика, учитывающая специфику штрих-кодов
- **mAP** (mean Average Precision) - точность на различных порогах IoU
- **Precision/Recall/F1** - классические метрики классификации
- **Processing Time** - время обработки для production мониторинга

### Кастомные метрики для штрих-кодов

Проект включает специализированные метрики, учитывающие особенности штрих-кодов:
- Проекционное IoU для вертикальных полос
- Устойчивость к шуму и искажениям
- Метрики читаемости штрих-кода

## 🔍 API документация

После запуска сервера автоматически доступна документация:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Основные endpoints

```http
POST /predict - Загрузка изображения и получение предсказаний
GET /health - Проверка состояния сервера
GET /metrics - Метрики производительности
```

### Пример использования API

```python
import requests

# Загрузка изображения
with open("barcode_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f},
        params={"confidence_threshold": 0.7}
    )

result = response.json()
print(f"Найдено штрих-кодов: {result['num_detections']}")
```

## ⚠️ Устранение неполадок

### Частые проблемы

#### 1. ModuleNotFoundError

```bash
# Решение 1: Установка пакета
pip install -e .

# Решение 2: Использование Poetry
poetry run python commands.py <команда>

# Решение 3: Настройка PYTHONPATH
export PYTHONPATH=$PWD/src  # Linux/Mac
set PYTHONPATH=%CD%\\src    # Windows
```

#### 2. Проблемы с Detectron2

```bash
# Для CPU версии
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# Для GPU версии (CUDA 11.1)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# Альтернативно - сборка из исходников
python scripts/install_detectron2.py
```

#### 3. Проблемы с DVC

```bash
# Переинициализация DVC
dvc init --force
dvc add data/raw
git add data/raw.dvc .gitignore

# Проверка игнорирования
git check-ignore -v data/raw
```

#### 4. MLflow connection issues

```bash
# Проверка доступности MLflow сервера
curl http://127.0.0.1:8080/health

# Запуск локального сервера
mlflow server --host 127.0.0.1 --port 8080 --backend-store-uri sqlite:///mlflow.db
```

## 🧪 Тестирование

### Запуск тестов

```bash
# Все тесты
pytest

# Тесты с покрытием
pytest --cov=src/barcode_segmentation --cov-report=html

# Только unit тесты
pytest -m unit

# Только integration тесты
pytest -m integration
```

### Проверка качества кода

```bash
# Запуск всех проверок
pre-commit run --all-files

# Отдельные инструменты
black src/
isort src/
flake8 src/
mypy src/
```

## 📚 Дополнительные ресурсы

### Документация

- [Hydra Configuration Guide](docs/hydra_guide.md)
- [MLflow Integration](docs/mlflow_guide.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Model Architecture](docs/model_architecture.md)

### Примеры использования

- [Training Custom Dataset](examples/train_custom_dataset.py)
- [Batch Inference](examples/batch_inference.py)
- [Model Optimization](examples/model_optimization.py)

### Научные статьи

- [Deep Dual Pyramid Network for Barcode Segmentation](http://arxiv.org/pdf/1807.11886.pdf)
- [Barcode Detection in Images](https://github.com/abbyy/barcode_detection_benchmark)

## 🤝 Участие в разработке

### Внесение изменений

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменений (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Создайте Pull Request

### Требования к коду

- Все изменения должны проходить pre-commit проверки
- Добавляйте тесты для новой функциональности
- Обновляйте документацию при необходимости
- Следуйте стилю кода проекта

## 🙏 Благодарности

- [Detectron2](https://github.com/facebookresearch/detectron2) за excellent framework
- [PyTorch Lightning](https://lightning.ai/) за упрощение ML экспериментов
- [Hydra](https://hydra.cc/) за мощную систему конфигурации
- Сообщество open-source за вдохновение и поддержку
