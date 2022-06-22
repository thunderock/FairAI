# @Filename:    glove.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/20/22 1:14 PM
import os
import numpy as np
import scipy.sparse
import struct

from models.model import Model

class Glove(Model):
    def __init__(self):
        pass

    def load_bin_vectors(self, embedding_path, vocab_size):
        n = os.path.getsize(embedding_path) // 8
        dim = (n - 2 * vocab_size) // (2 * vocab_size)
        W = np.zeros((vocab_size, dim))
        U = np.zeros((vocab_size, dim))
        b_w = np.zeros(vocab_size)
        b_u = np.zeros(vocab_size)
        with open(embedding_path, 'rb') as f:
            for i in range(vocab_size):
                for j in range(dim):
                    W[i, j] = np.float64(np.frombuffer(f.read(8), dtype=np.float64))
                b_w[i] = np.float64(np.frombuffer(f.read(8), dtype=np.float64))
            for i in range(vocab_size):
                for j in range(dim):
                    U[i, j] = np.float64(np.frombuffer(f.read(8), dtype=np.float64))
                b_u[i] = np.float64(np.frombuffer(f.read(8), dtype=np.float64))
        return W, b_w, U, b_u

    def load_model(self, embedding_dir, window_size=8):
        vocab_path = os.path.join(embedding_dir, 'vocab-C0-V20.txt')
        embedding_path = os.path.join(embedding_dir, 'vectors-C0-V20-W8-D25-R0.05-E15-S1.bin')
        vocab, ivocab = self.load_vocab(vocab_path)
        d = window_size
        V = len(vocab)
        W, b_w, U, b_u = self.load_bin_vectors(embedding_path, V)
        D = W.shape[1]
        return vocab, ivocab, W, b_w, U, b_u, D, d, vocab_path, embedding_path

    def load_vocab(self, path):
        str2idx, idx2str = dict(), []
        with open(path) as f:
            for (i, line) in enumerate(f):
                entry = line.split()
                str2idx[entry[0]] = (i, int(entry[1]))
                idx2str.append(entry[0])
        return str2idx, idx2str

    def read_cooc(self, line):
        entry = line.split()
        return (np.int32(entry[0]), np.int32(entry[1]), np.float64(entry[2]))


    def load_cooc(self, cooc_path, vocab_size):
        I, J, X = [], [], []
        file = open(cooc_path, 'rb')
        byte = file.read(16)

        while byte:
            i, j, x = struct.unpack('iid', byte)
            I.append(i - 1)
            J.append(j - 1)
            X.append(x)
            byte = file.read(16)
        file.close()
        return scipy.sparse.csc_matrix((X, (I, J)), shape=(vocab_size, vocab_size))

    def parse_coocs(self, text, vocab, window_size):
        words = text.split()
        i, j, l1, l2 = -2, -2, 0, -1
        net_offset = 0
        I, J, vals = [], [], []
        for l1, word in enumerate(words):
            if word not in vocab:
                continue
            i = vocab[word][0]
            l2 = l1
            net_offset = 0
            while net_offset < window_size:
                l2 -= 1
                if l2 < 0:
                    break
                if words[l2] not in vocab:
                    continue
                j = vocab[words[l2]][0]
                net_offset += 1
                vals.append(1.0/net_offset)
                I.append(i)
                J.append(j)
        I, J, vals = np.array(I), np.array(J), np.array(vals)
        return np.concatenate((I, J)), np.concatenate((J, I)), np.concatenate((vals, vals))

    def doc2cooc(self, text, vocab, window_size):
        V = len(vocab)
        I, J, vals = self.parse_coocs(text, vocab, window_size)
        return scipy.sparse.csc_matrix((vals, (I, J)), shape=(V, V))

    def Li(self, W, b_w, U, b_u, X, i):
        assert len(U.shape) == 2
        V, D = U.shape
        Xi = X[i, :].
    def f(self, x:np.float64, mx:np.float64=100., alpha:np.float64=0.75) -> np.float64:
        if x > mx:
            return np.float64(1.)
        return np.float64((x / mx) ** alpha)

    def compute_IF_deltas(self, document, vocab, ):
# "Model loading"
# In [1]: from models import glove
#
# In [2]: g = glove.Glove()
#
# In [3]: m = g.load_model('embeddings')
#
# In [4]: m[7]
# Out[4]: 8
#
# In [5]: m[6]
# Out[5]: 25
#
# In [6]: m[1][3]
# Out[6]: 'in'
#
# In [7]: m[1][2]
# Out[7]: 'and'
#
# In [8]: m[0]["is"]
# Out[8]: (5, 213234)


