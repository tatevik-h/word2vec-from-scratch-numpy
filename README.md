# word2vec-from-scratch-numpy
Pure NumPy implementation of  Word2Vec(Skip-Gram with Negative Sampling) including forward pass, loss computation, gradient derivation, and training loop.

## Run the Application

### Python
`pip install -r requirements.txt`

`python train.py`

### Docker
   `docker build -t word2vec-numpy .`
   
   `docker run --rm word2vec-numpy`

### Makefile
`make install`

`make train`
