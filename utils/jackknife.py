# @Filename:    jackknife.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/6/22 2:53 PM

import numpy as np
from utils import dataset, weat
from tqdm import tqdm, trange
from models.word2vec import Word2Vec

class JackKnife(object):
    def __init__(self, dataset):
        self.dataset = dataset
        assert self.dataset.stream is False, 'Streaming data not supported in JackKnife yet'

    def weat_scores(self):
        scores = np.empty((self.dataset.size, 7))
        size = self.dataset.size
        for i in trange(size):
            lines = self.dataset.lines[:i] + self.dataset.lines[i + 1:]
            model = Word2Vec(load=False)
            model.fit(dataset.TextCorpus(lines), workers=25)
            scorer = weat.WEAT(model)
            scores[i, :] = scorer.get_scores()
        return scores
