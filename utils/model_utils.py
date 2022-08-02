# @Filename:    model_utils.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/25/22 3:23 PM

import gensim

def identity(x):
    return x

LOAD_GENSIM_KEYED_VECTORS = gensim.models.KeyedVectors.load_word2vec_format

