# @Filename:    jackknife.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/6/22 2:53 PM

import numpy as np
from utils import dataset, weat
from tqdm import tqdm, trange
from models.word2vec import Word2Vec
import multiprocessing as mp


class JackKnife(object):
    def __init__(self, dataset):
        self.dataset = dataset
        assert self.dataset.stream is False, 'Streaming data not supported in JackKnife yet'

    @staticmethod
    def score_dataset_id(iid, instances):
        print(iid)
        lines = instances[:iid] + instances[iid + 1:]
        model = Word2Vec(load=False)
        model.fit(dataset.TextCorpus(lines), workers=1)
        scorer = weat.WEAT(model)
        return scorer.get_scores()

    def weat_scores(self):
        total = self.dataset.size
        threads = pool_size = 90
        # score_func = partial(JackKnife.score_dataset_id, instances=self.dataset.lines)
        final_result = np.zeros((total, 7))
        st = 0
        for i in trange(total // pool_size):
            st = pool_size * i
            pool = mp.Pool(processes=threads)
            result = pool.starmap(JackKnife.score_dataset_id, [(st + j, self.dataset.lines) for j in range(pool_size)])
            pool.close()
            pool.join()
            final_result[st: st + pool_size, :] = np.array(result)
        st += pool_size
        for i in trange(st, total):
            final_result[i, :] = np.array(JackKnife.score_dataset_id(i, self.dataset.lines))
        # print(final_result, st)
        return final_result
