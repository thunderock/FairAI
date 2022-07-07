# @Filename:    driver_torch.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        5/28/25 10:19 PM

import gc
from os.path import join as j
import numpy as np
from models.word2vec import Word2Vec
from tqdm import tqdm
from utils.dataset import Dataset
from utils.jackknife_torch import JackKnifeTorch
from models import fast_glove, word2vec
from torch.utils.data import DataLoader
import argparse
# ! wget -P /tmp/ http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# ! wget -P /tmp/ https://dumps.wikimedia.org/swwiki/latest/swwiki-latest-pages-articles.xml.bz2
DIMS = [25, 100]
WIKI = 'wiki'
EMBEDDINGS = 'embeddings'
OUTPUT = 'output'
WORD2VEC = 'word2vec'
GLOVE = 'glove'
DATASETS = [WIKI]
MODELS = {WORD2VEC: word2vec.Word2Vec, GLOVE: fast_glove.FastGlove}


def model(model_name, dim): return MODELS[model_name](load=True, dim=dim)


def dataset(dataset_file, dataset_type):
    if dataset_type == WIKI:
        return Dataset(dataset_file)
    else:
        assert False, 'Dataset type not supported yet!'

def file_name(outfile):
    return outfile

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_type', type=str, default=WIKI, help='Dataset to use')
parser.add_argument('--model_name', type=str, default=GLOVE, help='Model to use')
parser.add_argument('--threads', type=int, default=55, help='Number of threads to use')
parser.add_argument('--dataset_file', type=str, required=True, help='Dataset file to use')
parser.add_argument('--dim', type=int, default=100, help='Dimension of the model')
parser.add_argument('--outfile', type=str, default='output.npy', help='Output file')
args = parser.parse_args()
# print(ds.lines)
m = model(args.model_name, args.dim)
ds = dataset(args.dataset_file, args.dataset_type)
jk = JackKnifeTorch(dataset=ds, model=m)
total = len(jk)
total = ds.size
threads = args.threads
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
        np.save(file_name(args.outfile), scores)
    del indices
    gc.collect()
    status_loop.set_description('Processing batch %d' % i)

np.save(file_name(args.outfile), scores)




