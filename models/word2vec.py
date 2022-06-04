# @Filename:    word2vec.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        5/28/22 12:01 AM

import gensim
from model import Model


class Word2Vec(Model):

    def __init__(self, load=False, window_size=10, min_count=5, dim=100,path=None):
        super().__init__(dim, load, window_size, min_count, path)
        self.window_size = window_size
        self.min_count = min_count
        self.dim = dim
        self._model = None
        if load:
            self.load(path)

    def fit(self, dataset, workers=4):
        self._model = gensim.models.Word2Vec(dataset.lines, window=self.window_size, min_count=self.min_count,
                                             workers=workers, vector_size=self.dim).wv
        return self._model

    def save(self, path):
        assert self._model is not None, 'Model not fitted yet'
        self._model.save(path)

    def load(self, path):
        self._model = gensim.models.KeyedVectors.load(path)

    def transform(self, words):
        words = [w for w in words if w in self._model]
        return self._model[words]


