# @Filename:    model.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/3/22 11:04 PM

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
        assert False, 'Not implemented'

    def transform(self, words, WV):
        assert False, 'Not implemented'