"""
Скрипт для настройки окружения.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str):
    """Выполняет команду и обрабатывает ошибки."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ {description} выполнено успешно")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Ошибка при {description.lower()}: {e}")
        print(f"Вывод: {e.stdout}")
        print(f"Ошибки: {e.stderr}")
        return False


def setup_environment():
    """Настраивает окружение для проекта."""
    print("="*60)
    print("НАСТРОЙКА ОКРУЖЕНИЯ BARCODE SEGMENTATION")
    print("="*60)

    # Проверяем наличие Poetry
    print("\nПроверяем Poetry...")
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        print("✓ Poetry найден")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Poetry не найден. Установите Poetry:")
        print("curl -sSL https://install.python-poetry.org | python3 -")
        return False

    # Устанавливаем зависимости
    if not run_command("poetry install", "Установка зависимостей"):
        return False

    # Настраиваем pre-commit
    if not run_command("poetry run pre-commit install", "Настройка pre-commit"):
        return False

    # Инициализируем DVC
    if not Path(".dvc").exists():
        if not run_command("poetry run dvc init", "Инициализация DVC"):
            return False

    # Создаем необходимые директории
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models",
        "plots",
        "outputs"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Создана директория: {directory}")

    print("\n" + "="*60)
    print("НАСТРОЙКА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*60)
    print("\nСледующие шаги:")
    print("1. Скопируйте данные в data/raw/")
    print("2. Запустите: python commands.py preprocess")
    print("3. Запустите: python commands.py train")
    print()
    return True


if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)
