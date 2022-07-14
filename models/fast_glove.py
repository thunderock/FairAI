# @Filename:    fast_glove.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/24/22 1:46 PM
import os
import numpy as np
from utils import glove
from models.model import Model
import pickle as pkl


class FastGlove(Model):
    def __init__(self, embedding_dir="embeddings", dim=100, load=True):
        # dim fixed comes from trained model
        # need to fix this, written specifically for word2vec
        assert load, "load must be true"
        super().__init__(dim=dim, load=load, window_size=8, min_count=10, path=embedding_dir)

    def load(self, path):
        self.g = glove.Glove()
        self.path = path
        self.M = self.g.load_model(path)
        V = len(self.M.vocab)
        cooc_path = os.path.join(path, "cooc-C0-V10-W8.bin")
        self.X = self.g.load_cooc(cooc_path, V)
        self.weat_words = pkl.load(open(os.path.join(path, '..', "weat/words.pkl"), "rb"))

    def fit(self, iid, dataset, workers=1):
        document = dataset[iid]
        deltas = self.g.compute_IF_deltas(document, self.M, self.X, self.weat_words)
        return self.g.get_new_W(self.M, deltas)

    def transform(self, words, W=None):
        if W is None:
            W = self.M.W
        indices = [self.M.vocab[w][0] for w in words if w in self.M.vocab]
        ret = np.empty((len(indices), self.M.D))
        for i, idx in enumerate(indices):
            ret[i, :] = W[idx]
        return ret




