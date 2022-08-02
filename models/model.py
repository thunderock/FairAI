# @Filename:    model.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/3/22 11:04 PM
import numpy as np


class Model(object):
    def __init__(self, dim=100, load=False, window_size=10, min_count=5, path=None):
        self.window_size = window_size
        self.min_count = min_count
        self.dim = dim
        self._model = None
        if load:
            self.load(path)

    def fit(self, iid, dataset, workers=4):
        assert False, 'Not implemented'

    def save(self, path):
        assert False, 'Not implemented'

    def load(self, path):
        assert False, 'Not implemented, should take path and load model'

    def transform(self, words, WV=None):
        """
        to call this WV should be None, attempts to be called from custom models for less work, uses _model attribute
        :param words:
        :param WV:
        :return:
        """
        indices = []
        not_found = 0
        for w in words:
            if w in self._model:
                indices.append(self._model[w])
            else:
                not_found += 1

        if not_found > 0:
            print("{} words not found in model".format(not_found))
        return np.array(indices)