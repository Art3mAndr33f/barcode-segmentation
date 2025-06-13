# Makefile для проекта barcode-detection-mlops

.PHONY: help install install-dev setup train infer export lint format test clean docker-build docker-run

help:
	@echo "Доступные команды:"
	@echo "  install      - Установка зависимостей"
	@echo "  install-dev  - Установка dev зависимостей" 
	@echo "  setup        - Полная настройка окружения"
	@echo "  train        - Обучение модели"
	@echo "  infer        - Запуск инференса"
	@echo "  export       - Экспорт модели в ONNX/TensorRT"
	@echo "  lint         - Проверка качества кода"
	@echo "  format       - Форматирование кода"
	@echo "  test         - Запуск тестов"
	@echo "  clean        - Очистка временных файлов"

install:
	poetry install --no-dev

install-dev:
	poetry install
	poetry run pre-commit install

setup: install-dev
	@echo "Настройка DVC..."
	poetry run dvc init --no-scm
	poetry run dvc remote add -d myremote gdrive://YOUR_GOOGLE_DRIVE_FOLDER_ID
	@echo "Настройка завершена!"

train:
	poetry run python src/commands.py +command=train

infer:
	poetry run python src/commands.py +command=infer

export:
	poetry run python src/commands.py +command=export

download-data:
	poetry run python src/commands.py +command=download_data

lint:
	poetry run pre-commit run --all-files

format:
	poetry run black src/
	poetry run isort src/

test:
	poetry run pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	rm -rf outputs/
	rm -rf inference_results/

docker-build:
	docker build -t barcode-detection-mlops .

docker-run:
	docker run --gpus all -p 8000:8000 barcode-detection-mlops

# DVC команды
dvc-add-data:
	poetry run dvc add data/
	git add data.dvc .gitignore
	git commit -m "Add data to DVC"

dvc-push:
	poetry run dvc push

dvc-pull:
	poetry run dvc pull

# MLflow команды  
mlflow-ui:
	poetry run mlflow ui --host 127.0.0.1 --port 8080
