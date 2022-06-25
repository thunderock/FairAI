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
    def __init__(self, embedding_dir):
        # dim fixed comes from trained model
        # need to fix this, written specifically for word2vec
        super().__init__(dim=25, load=True, window_size=8, min_count=None, path=embedding_dir)

    def load(self, path):
        self.g = glove.Glove()
        self.path = path
        self.M = self.g.load_model(path)
        V = len(self.M.vocab)
        cooc_path = os.path.join(path, "cooc-C0-V20-W8.bin")
        self.X = self.g.load_cooc(cooc_path, V)
        self.weat_words = pkl.load(open("weat/words.pkl", "rb"))


    def fit(self, iid, dataset, workers=1):
        document = dataset[iid]
        y = self.g.compute_IF_deltas(document, self.M, self.X)




