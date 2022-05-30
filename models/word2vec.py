# @Filename:    word2vec.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        5/28/22 12:01 AM

import gensim


class Word2Vec(object):

    def __init__(self, lines=None):
        self.lines = lines
        self._model = None

    def fit(self, window_size=10, min_count=5, workers=4, dim=100):
        self._model = gensim.models.Word2Vec(self.lines, window=window_size, min_count=min_count,
                                                workers=workers, vector_size=dim).wv
        return self._model

    def save(self, path):
        assert self._model is not None, 'Model not fitted yet'
        self._model.save(path)

    def load(self, path):
        self._model = gensim.models.KeyedVectors.load(path)

    def get_word_vectors(self, words):
        words = [w for w in words if w in self._model]
        return self._model[words]



