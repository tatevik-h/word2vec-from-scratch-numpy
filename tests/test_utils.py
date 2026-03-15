import numpy as np

from utils import tokenize, build_vocab, generate_pairs, negative_sampling


def test_tokenize(tmp_path):
    file = tmp_path / "corpus.txt"
    file.write_text("hello world hello")

    tokens = tokenize(str(file))

    assert tokens == ["hello", "world", "hello"]


def test_build_vocab():
    tokens = ["a", "b", "a"]

    vocab, word_to_id, id_to_word = build_vocab(tokens)

    assert len(vocab) == 2
    assert word_to_id["a"] != word_to_id["b"]
    assert id_to_word[word_to_id["a"]] == "a"


def test_generate_pairs():
    tokens = ["i", "love", "nlp"]
    vocab, word_to_id, _ = build_vocab(tokens)

    pairs = generate_pairs(tokens, word_to_id, window_size=1)

    assert len(pairs) > 0
    assert isinstance(pairs[0], tuple)


def test_negative_sampling():
    vocab_size = 10
    context_id = 3

    negatives = negative_sampling(vocab_size, context_id, 5)

    assert len(negatives) == 5
    assert context_id not in negatives
