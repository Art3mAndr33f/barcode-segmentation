.PHONY: install train evaluate deploy clean lint test 

install: 
  poetry install 
  poetry run pre-commit install 

setup-data: 
  poetry run python scripts/prepare_data.py 

train: 
  poetry run python scripts/train.py 

evaluate: 
  poetry run python scripts/evaluate.py 

lint: 
  poetry run black src/ scripts/ tests/ 
  poetry run isort src/ scripts/ tests/ 
  poetry run flake8 src/ scripts/ tests/ 

test: 
  poetry run pytest tests/ 

clean: 
  find . -type f -name "*.pyc" -delete 
  find . -type d -name "pycache" -delete 
  rm -rf .pytest_cache 

docker-build: 
  docker build -t barcode-detection . 

docker-run: 
  docker run -p 8000:8000 barcode-detection 

help: 
  @echo "����㯭� �������:" 
  @echo "  install     - ��⠭����� ����ᨬ���" 
  @echo "  setup-data  - �����⮢��� �����" 
  @echo "  train       - ������ ������" 
  @echo "  evaluate    - �業��� ������" 
  @echo "  lint        - �஢���� ���" 
  @echo "  test        - �������� ����" 
  @echo "  clean       - ������ �६���� 䠩��" 
