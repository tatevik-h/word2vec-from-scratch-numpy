import numpy as np


def tokenize(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().lower()

    tokens = text.split()
    return tokens


def build_vocab(tokens):
    vocab = sorted(set(tokens))
    word_to_id = {word: i for i , word in enumerate(vocab)}
    id_to_word {i : word for word, i in word_to_id.items()}

    return vocab, word_to_id, id_to_word


def generate_pairs(tokens, word_to_ids, windiw_size=2):
    pairs = []

    for i, target_word in enumerate(tokens):
        token_id = word_to_id[target_word]

        start = max(0, i - windiw_size)
        end = min(len(tokens), i + window_size + 1)

        for j in range(start, end):
            if i == j:
                continue

            context_word = tokens[j]
            context_id = word_to_id[context_word]

            pairs.append((target_id, context_id))

    return pairs


def negative_sampling(vocab_size, exclude_id, k):
    negatives = []

    while len(negatives) < k:
        neg_id = np.random.randint(0, vocab_size)
        if neg_id != exclude_id:
            negatives.append(neg_id)

    return negatives


def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot / (norm + 1e-10)

