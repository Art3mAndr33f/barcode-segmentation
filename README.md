# Barcode Segmentation MLOps Project

MLOps проект для автоматической сегментации штрих-кодов с использованием Detectron2 и современных инструментов машинного обучения.

## 🎯 Описание проекта

Проект реализует полный pipeline машинного обучения для задачи сегментации штрих-кодов:
- Использует архитектуру Mask R-CNN на базе ResNet-50
- Поддерживает версионирование данных с DVC
- Отслеживание экспериментов через MLflow
- Автоматизированный training pipeline с PyTorch Lightning
- Экспорт моделей в ONNX и TensorRT для продакшена

## 🛠 Технологический стек

- **Модель**: Detectron2 (Mask R-CNN + ResNet-50)
- **Framework**: PyTorch Lightning
- **Управление зависимостями**: Poetry
- **Версионирование данных**: DVC
- **Конфигурация**: Hydra
- **Эксперименты**: MLflow
- **Качество кода**: pre-commit, black, isort, flake8
- **Экспорт**: ONNX, TensorRT

## 📁 Структура проекта

```
barcode_segmentation/
├── data/
│   ├── raw/                    # Исходные изображения (.jpg) и аннотации (.jpg.json)
│   ├── processed/              # Обработанные данные
│   └── interim/                # Промежуточные данные
├── src/barcode_segmentation/   # Исходный код проекта
│   ├── data/                   # Модули работы с данными
│   ├── models/                 # Модели и training
│   └── utils/                  # Вспомогательные функции
├── configs/                    # Конфигурационные файлы Hydra
├── scripts/                    # Скрипты установки и настройки
├── commands.py                 # CLI команды проекта
├── pyproject.toml             # Конфигурация Poetry
└── dvc.yaml                   # Pipeline DVC
```

## 🚀 Быстрый старт

### 1. Установка окружения

```bash
# Клонирование репозитория
git clone https://github.com/YOUR_USERNAME/barcode_segmentation.git
cd barcode_segmentation

# Установка зависимостей
poetry install

# Для Windows: специальная установка detectron2
python install_detectron2.py
```

### 2. Подготовка данных

```bash
# Поместите ваши данные в data/raw/:
# - .jpg файлы (изображения)
# - .jpg.json файлы (аннотации)

# Инициализация DVC (если не настроен)
dvc init
dvc add data/raw
git add data/raw.dvc data/.gitignore
git commit -m "Add raw data to DVC"
```

### 3. Запуск pipeline

```bash
# Установка пакета в editable режиме
pip install -e .

# Предобработка данных
python commands.py preprocess

# Обучение модели
python commands.py train

# Инференс на новых данных
python commands.py infer
```

## 📊 Мониторинг экспериментов

Для отслеживания экспериментов запустите MLflow UI:

```bash
mlflow server --host 127.0.0.1 --port 8080
```

Веб-интерфейс будет доступен по адресу: http://127.0.0.1:8080

## 🔧 Основные команды

| Команда | Описание |
|---------|----------|
| `python commands.py preprocess` | Предобработка и валидация данных |
| `python commands.py train` | Обучение модели с логированием в MLflow |
| `python commands.py infer` | Инференс на новых данных |
| `python commands.py convert_to_onnx` | Конвертация модели в ONNX |
| `python commands.py convert_to_tensorrt` | Оптимизация для TensorRT |

## ⚠️ Устранение неполадок

### Ошибка ModuleNotFoundError

```bash
# Решение 1: Установка пакета
pip install -e .

# Решение 2: Использование Poetry
poetry run python commands.py <команда>

# Решение 3: Настройка PYTHONPATH
set PYTHONPATH=%CD%\src  # Windows
export PYTHONPATH=$PWD/src  # Linux/Mac
```

### Проблемы с DVC

```bash
# Если DVC файл игнорируется Git
# Проверьте .gitignore и удалите строки типа data/ или data/raw/
git check-ignore -v data/raw

# Переинициализация DVC
dvc add data/raw
git add data/raw.dvc data/.gitignore
```

### Установка Detectron2 на Windows

```bash
# Используйте специальный скрипт
python install_detectron2.py

# Или ручная установка
poetry shell
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html
```

## 📈 Метрики и оценка

Проект отслеживает следующие метрики:
- **IoU (Intersection over Union)**: стандартная метрика сегментации
- **Modified IoU**: кастомная метрика для специфики штрих-кодов
- **Loss**: функция потерь во время обучения
- **mAP**: средняя точность на разных порогах IoU

## 🚀 Продакшен деплоймент

### Экспорт модели

```bash
# ONNX для кроссплатформенности
python commands.py convert_to_onnx

# TensorRT для оптимизации на NVIDIA GPU
python commands.py convert_to_tensorrt
```

### Docker контейнеризация

```bash
# Сборка образа
docker build -t barcode-segmentation .

# Запуск контейнера
docker run -p 8000:8000 barcode-segmentation
```