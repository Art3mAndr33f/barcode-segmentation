# Базовый образ с CUDA поддержкой
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Устанавливаем переменные окружения
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Обновляем систему и устанавливаем зависимости
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Создаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock ./

# Настраиваем Poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev

# Копируем исходный код
COPY . .

# Устанавливаем Detectron2
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Создаем необходимые директории
RUN mkdir -p data/raw data/processed models plots

# Устанавливаем права доступа
RUN chmod +x commands.py

# Экспонируем порты для MLflow и других сервисов
EXPOSE 8080 8000 8001 8002

# Устанавливаем переменные окружения для MLOps
ENV MLFLOW_TRACKING_URI=http://localhost:8080
ENV HYDRA_FULL_ERROR=1

# Команда по умолчанию
CMD ["python3", "commands.py", "--help"]
