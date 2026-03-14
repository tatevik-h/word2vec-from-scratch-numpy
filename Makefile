.PHONY: install train train docer-build docker-run clean

install:
	pip install -r requirements.txt

train: 
	python train.py

docker-build:
	python build -t word2vec-numpy .

docker-run:
	docker run --rm word2vec-numpy

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
