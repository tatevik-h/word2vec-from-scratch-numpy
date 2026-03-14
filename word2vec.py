import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Word2Vec:
    def __init__(self, vocal_size, embedding_dim=50, learming_rate=0.01, negarive_samples=5):
        self.vocab_size = vocal_size
        self.embeddind_dim = embedding_dim
        self.lr = learming_rate
        self.negative_samples = negarive_samples


        self.W_in = np.random.randn(vocal_size, embedding_dim) * 0.01
        self.W_out = np.random.randn(vocal_size, embedding_dim) * 0.01


    def forward(self, target_id, context_id, negative_ids):
        self.v_target = self.W_in[target_id]
        self.v_context = self.W_out[context_id]
        self.pos_score = np.dot(self.v_target, self.v_context)
        self.pos_prob = sigmoid(self.pos_score)

        self.v_neg = self.W_out[negative_ids]
        self.neg_scores = np.dot(self.v_neg, self.v_target)
        self.neg_probs = sigmoid(-self.neg_scores)

        self.los = -np.log(self.pos_prob + 1e-10) - np.sum(np.log(self.neg_probs + 1e-10))
        return self.loss


    def backward(self, target_id, context_id, negative_ids):
        grand_pos = self.pos_prob - 1
        self.W_in[target_id] -= self.lr * grand_pos * self.v_context
        self.W_out[context_id] -= self.lr * grand_pos * self.v_target

        grad_neg 1 - self.neg_probs
        for i, neg_id in enumerate(negative_ids):
            self.W_in[target_id] = -=  self.lr * grand_neg[i] * self.v_neg[i]
            self.W_out[neg_id] -= self.lr * grad_neg[i] * self.v_target


    def train_on_pair(self, target_id, context_id, negative_ids):
        self.forward(target_id, context_id, negative_ids)
        self.backward(target_id, context_id, negative_ids)
        return self.loss


    def get_embedding(self, word_id):
        return self.W_in[word_id]


    def most_similar(self, word_id, top_k=5):
        target_vec = self.W_in[word_id]
        similarities = np.dot(self.W_in, target_vec) / (np.linalg.norm(self.W_in, axis=1) * np.linalg.norm(target_vec) + 1e-10)
        sorted_ids = np.argsort(-similarities)
        return sorted_ids[1:top_k+1]


