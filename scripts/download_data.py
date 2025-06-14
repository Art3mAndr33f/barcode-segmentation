"""
Скрипт для загрузки данных штрих-кодов.
"""

from pathlib import Path


def download_data():
    """
    Функция для загрузки данных.

    Поскольку данные находятся на локальной машине пользователя,
    этот скрипт предоставляет инструкции по их размещению.
    """
    print("="*60)
    print("ИНСТРУКЦИИ ПО НАСТРОЙКЕ ДАННЫХ")
    print("="*60)
    print()
    print("Поскольку ваши данные находятся на локальной машине,")
    print("выполните следующие шаги:")
    print()
    print("1. Создайте директорию data/raw/ в корне проекта:")
    print("   mkdir -p data/raw")
    print()
    print("2. Скопируйте ваши изображения и JSON файлы в data/raw/")
    print("   Структура должна быть:")
    print("   data/raw/")
    print("   ├── image1.jpg")
    print("   ├── image1.jpg.json")
    print("   ├── image2.jpg")
    print("   ├── image2.jpg.json")
    print("   └── ...")
    print()
    print("3. Убедитесь что JSON файлы содержат аннотации в правильном формате:")
    print("   {")
    print('     "size": [width, height],')
    print('     "objects": [')
    print("       {")
    print('         "data": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]')
    print("       }")
    print("     ]")
    print("   }")
    print()
    print("4. После размещения данных запустите предобработку:")
    print("   python commands.py preprocess")
    print()
    print("5. Затем можете запустить тренировку:")
    print("   python commands.py train")
    print()
    print("="*60)


if __name__ == "__main__":
    download_data()
