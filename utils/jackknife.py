# @Filename:    jackknife.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/6/22 2:53 PM

import numpy as np
from utils import dataset, weat
from tqdm import tqdm, trange
from models.word2vec import Word2Vec
from functools import partial

class JackKnife(object):
    def __init__(self, dataset):
        self.dataset = dataset
        assert self.dataset.stream is False, 'Streaming data not supported in JackKnife yet'

    @staticmethod
    def score_dataset_id(id, instances):
        lines = instances[:id] + instances[id + 1:]
        model = Word2Vec(load=False)
        model.fit(dataset.TextCorpus(lines), workers=1)
        scorer = weat.WEAT(model)
        return scorer.get_scores()

    def weat_scores(self):
        size = self.dataset.size
        import multiprocessing as mp
        pool = mp.Pool(processes=20)
        score_func = partial(JackKnife.score_dataset_id, instances=self.dataset.lines)
        result = pool.map(score_func, trange(size))
        pool.close()
        pool.join()
        return result