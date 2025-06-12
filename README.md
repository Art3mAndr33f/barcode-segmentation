# barcode-segmentation

- Предварительные условия:

1) **Google Colab:**  Этот код предназначен для запуска в Google Colab.

2) **PyTorch:** Detectron2 требует PyTorch.

3) **Detectron2:** Этот проект использует Detectron2.

4) **OpenCV (cv2):**  Используется для обработки изображений.

5) **JSON:**  Используется для чтения файлов аннотаций.

6) **CUDA (опционально):** Для более быстрой тренировки рекомендуется использовать GPU-среду выполнения.

- Setup:
    1.  Клонируйте репозиторий: git clone <repo_url>
    2.  Установите Poetry: pip install poetry
    3.  Перейдите в директорию проекта: cd barcode-segmentation
    4.  Создайте виртуальное окружение и установите зависимости: poetry install
    5.  Активируйте виртуальное окружение: poetry shell
    6.  Установите pre-commit: pip install pre-commit
    7.  Установите DVC: pip install dvc
    8.  Инициализируйте DVC: dvc init (если вы используете удалённое хранилище DVC, настройте его здесь)
    9.  Установите pre-commit хуки: pre-commit install

- Train:
    1.  Загрузите данные (если используете локальный источник): python barcode_segmentation/data/datasets.py --download (если используете DVC, пропустите этот шаг).
    2.  Запустите тренировку: python barcode_segmentation/train.py --config-path configs --config-name train

- Production preparation:
    1.  Конвертируйте модель в ONNX (включено в train.py или отдельный скрипт).
    2.  Конвертируйте модель в TensorRT (скрипт onnx_to_tensorrt.sh или python скрипт).
    3.  Артефакты для продакшена: model.onnx, model.trt, configs/infer.yaml.

- Infer:
    1.  Подготовьте данные для инференса (изображения).
    2.  Запустите инференс: python barcode_segmentation/infer.py --config-path configs --config-name infer --image-path <путь_к_изображению>
