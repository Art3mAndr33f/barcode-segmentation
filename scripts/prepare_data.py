import os
import shutil
from pathlib import Path

def prepare_data():
    """Подготовка данных для обучения"""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    # Создание директорий
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "train").mkdir(exist_ok=True)
    (processed_dir / "val").mkdir(exist_ok=True)
    (processed_dir / "test").mkdir(exist_ok=True)
    
    print("Данные подготовлены!")

if __name__ == "__main__":
    prepare_data()
EOF