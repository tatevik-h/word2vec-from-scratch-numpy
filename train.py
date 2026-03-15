import argparse

from word2vec import Word2Vec
from utils import negative_sampling, tokenize, build_vocab, generate_pairs
from config import *
from logger import get_logger


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Word@Vec model")

    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--embedding_dim", type=int, default=50, help="Embedding dimension")
    parser.add_argument("--window_size", type=int, default=2, help="Context window size")

    return parser.parse_args()


def main():
    args = parse_args()

    epochs = args.epochs
    embedding_dim = args.embedding_dim
    window_size = args.window_size


    logger.info("Starting Word2Vec training")
    tokens = tokenize(CORPUS_PATH)
    
    vocab, word_to_id, id_to_word = build_vocab(tokens)

    pairs = generate_pairs(tokens, word_to_id, window_size)

    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Training pairs: {len(pairs)}")

    model = Word2Vec(vocab_size=len(vocab), embedding_dim=embedding_dim, learning_rate=LEARNING_RATE, negative_samples=NEGATIVE_SAMPLES)

    for epoch in range(epochs):
        total_loss = 0
        for target_ids, context_id in pairs:
            negative_ids = negative_sampling(len(vocab), context_id, NEGATIVE_SAMPLES)
            loss = model.train_on_pair(target_id, context_id, negative_ids)
            total_loss += loss

        logger.info(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")


    word = list(word_to_id.keys())[0]
    word_id = word_to_id[word]

    logger.info(f"Example word: {word}")

    similar_ids = model.most_similar(word_id)

    logger.info("Most similar words:")

    for i in similar_ids:
        logger.info(id_to_word[i])


if __name__ == "__main__":
    main()

