from os.path import join as j
from itertools import product
import numpy as np
from torch.utils.data import DataLoader
from models import word2vec, fast_glove
from utils import dataset, jackknife_torch

DIMS = [25, 100]
WIKI = 'wiki'
EMBEDDINGS = 'embeddings'
OUTPUT = 'output'
WORD2VEC = 'word2vec'
GLOVE = 'glove'
DATASETS = [WIKI]
MODELS = {WORD2VEC: word2vec.Word2Vec, GLOVE: fast_glove.FastGlove}
DATA_SRC = {
    WIKI: '../simplewiki-20171103-pages-articles-multistream.xml.bz2',
    EMBEDDINGS: 'embeddings',
    OUTPUT: 'data'
}

embeddings_params = {
    "threads": 55,
    "dim": 100,
    "embedding_dir": DATA_SRC[EMBEDDINGS],
    "output_dir": DATA_SRC[OUTPUT],
    "window_size": 8,
    "min_count": 10,
    "corpus_id": 0,
}

VOCAB_FILE = j("{embedding_dir}",'vocab-C0-V{min_count}.txt')
EMBEDDING_FILE = j("{embedding_dir}", 'vectors-C0-V{min_count}-W{window_size}-D{dim}-R0.05-E50-S1.bin'),
COOC_PATH = j("{embedding_dir}",'cooc-C0-V{min_count}-W{window_size}.bin')
SCORES_OUTPUT = j("{output_dir}", 'weat_scores_{dim}.npy')

rule calculate_glove_weat_scores_100:
    input:
        dataset = DATA_SRC[WIKI],
        vocab_file = expand(VOCAB_FILE, **embeddings_params),
        embedding_file = expand(EMBEDDING_FILE, **embeddings_params),
        cooc_path = expand(COOC_PATH, **embeddings_params),
    threads: 55
    output:
        out = expand(SCORES_OUTPUT, **embeddings_params)
    shell:
        "python driver_torch.py --dataset_type={WIKI} --dataset_file={input.dataset} --model_name={GLOVE} --dim=100 --outfile={output.out} --threads={threads}"

