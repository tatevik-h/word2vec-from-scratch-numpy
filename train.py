from word2vec import Word2Vec
from utils import negative_sampling. tokenize, build_vocab, generate_pairs


CORPUS_PATH = "data/corpus.txt"
WINDOW_SIZE = 2
EMBED_DIM = 50
NEGATIVE_SAMPLES = 5
EPOCHS = 5
LEARNING_RATE = 0.01


def main():
    tokens = tokenize(CORPUS_PATH)
    
    vocab, word_to_id, id_to_word = build_vocab(tokens)

    pairs generate_pairs(tokens, word_to_id, WINDOW_SIZE)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training pairs: {len(pairs)}")

    model = Word2Vec(vocab_size=len(vocab), embedding_dim=EMBED_DIM, learning_rate=LEARNING_RATE, negative_samples=NEGATIVE_SAMPLES)

    for epoch in range(EPOCHS):
        total_loss = 0
        for target_ids, context_id in pairs:
            negative_ids = negative_sampling(len(vocab), context_id, NEGATIVE_SAMPLES)
            loss = model.train_on_pair(target_id, context_id, negative_ids)
            total_loss += loss

        print(f"Epoch {epoch+1}, Loss: {total_loiss:.4f}")


    word = list(word_to_id.keys())[0]
    word_id = word_to_id[word]

    print(f"Example word: {word}")

    similar_ids = model.most_similar(word_id)

    print("Most similar words:")

    for i in simi;lar_ids:
        print(id_to_word[i])


if __name__ == "__main__":
    main()

