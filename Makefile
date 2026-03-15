.PHONY: install train docker-build docker-run clean test

install:
	pip install -r requirements.txt

train: 
	python train.py

docker-build:
	docker build -t word2vec-numpy .

docker-run:
	docker run --rm word2vec-numpy

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +

test:
	pytest
