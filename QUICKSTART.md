# Быстрый старт - Barcode Segmentation

## Установка окружения

```bash
cd barcode_segmentation
python scripts/setup_environment.py
```

## Подготовка данных

1. Скопируйте ваши изображения (.jpg) в `data/raw/`
2. Скопируйте соответствующие JSON аннотации (.jpg.json) в `data/raw/`

Формат JSON:
```json
{
  "size": [width, height],
  "objects": [
    {
      "data": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ]
}
```

## Запуск пайплайна

```bash
# 1. Предобработка данных
python commands.py preprocess

# 2. Обучение модели
python commands.py train

# 3. Инференс
python commands.py infer
```

## Мониторинг (в отдельном терминале)

```bash
mlflow server --host 127.0.0.1 --port 8080
# Откройте http://127.0.0.1:8080 в браузере
```

## Продакшн деплоймент

```bash
# Конвертация в ONNX
python commands.py convert_to_onnx

# Конвертация в TensorRT (опционально)
python commands.py convert_to_tensorrt

# Настройка Triton Inference Server
python commands.py setup_triton
```

## Полезные команды

```bash
# Справка по командам
python commands.py --help

# Изменение параметров
python commands.py train training.max_epochs=20

# Проверка кода
pre-commit run --all-files

# Тестирование
pytest tests/
```
