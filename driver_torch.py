# @Filename:    driver_torch.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        5/28/25 10:19 PM

import gc
import numpy as np
from models.word2vec import Word2Vec
from tqdm import tqdm
from utils.dataset import Dataset
from utils.jackknife_torch import JackKnifeTorch
from models.fast_glove import FastGlove
from torch.utils.data import DataLoader
# ! wget -P /tmp/ http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# ! wget -P /tmp/ https://dumps.wikimedia.org/swwiki/latest/swwiki-latest-pages-articles.xml.bz2

# load file
# file = open("../enwik9.txt", "r")
# lines = file.readlines()
# sents = [word_tokenize(line.lower()) for line in tqdm(lines)]

# train the model
# file = '/tmp/swwiki-latest-pages-articles.xml.bz2'
# dataset = object
# ds = Dataset(file)
# model = Word2Vec(load=False)
# model.fit(ds, workers=6)
# model.save("../word2vec.model")


# model = Word2Vec(load=True, path='../word2vec.model')
#
#
# weat = WEAT(model, 'weat/weat.json')
#
# weat_scores = weat.get_scores()
# print(weat_scores)
model = Word2Vec
model = FastGlove
ds = Dataset('../simplewiki-20171103-pages-articles-multistream.xml.bz2')
# print(ds.lines)
jk = JackKnifeTorch(ds, model)
total = len(jk)
total = ds.size
threads = 55
loops = total // threads + 1
loader = DataLoader(jk, batch_size=threads, shuffle=False)
scores = np.zeros((total, 7))
print(loops)
status_loop = tqdm(loader, total=loops)

for i, scores_ in enumerate(status_loop):
    indices = scores_[1]
    for ix, idx in enumerate(indices):
        if idx < total:
            scores[idx, :] = scores_[0][ix]
    if i == loops:
        break
    if i % 100:
        np.save('scores.npy', scores)
    del indices
    gc.collect()
    status_loop.set_description('Processing batch %d' % i)

np.save('scores.npy', scores)




