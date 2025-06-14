FROM python:3.8-slim

# Установка системных зависимостей для OpenCV, Detectron2 и компиляции
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    cmake \
    libopencv-dev \
    python3-opencv \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка Poetry
RUN pip install --no-cache-dir poetry==1.5.1

# Настройка Poetry
RUN poetry config virtualenvs.create false

# Установка рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY pyproject.toml poetry.lock* ./

# Установка зависимостей проекта
RUN poetry install --no-dev --no-root --no-interaction

# Установка специальных зависимостей 
# Detectron2 для CPU (можно изменить на GPU, если нужно)
RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html

# Копирование файлов проекта
COPY . .

# Устанавливаем пакет проекта
RUN pip install -e .

# Установка переменных окружения
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Открываем порт для сервера FastAPI
EXPOSE 8000

# Точка входа - запуск inference сервера
ENTRYPOINT ["python", "-m", "barcode_segmentation.deployment.inference_server"]

# Команда по умолчанию - параметры запуска сервера
CMD ["--host", "0.0.0.0", "--port", "8000"]