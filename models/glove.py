# @Filename:    glove.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/20/22 1:14 PM
import os
import numpy as np
import scipy.sparse
import struct

INT_INF = np.iinfo(np.int32).max
INT_MIN = np.iinfo(np.int32).min

from models.model import Model

class GloveWrapper(object):
    def __init__(self):
        self.vocab, self.ivocab, self.W, self.b_w, self.U, self.b_u, self.D, self.d, self.vocab_path, self.embedding_path = [None] * 10


class Glove(Model):
    def __init__(self):
        pass
    #
    # def non_zero_values_and_indices(self, X:scipy.sparse.csc_matrix, row_index:int=INT_INF, col_index:int=INT_INF) -> (np.array, np.array):
    #     values = X[row_index if row_index != INT_INF else , col_index].nonzero()[0]



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
        model = GloveWrapper()
        model.vocab, model.ivocab, model.W, model.b_w, model.U, model.b_u, model.D, model.d, model.vocab_path, model.embedding_path = vocab, ivocab, W, b_w, U, b_u, D, d, vocab_path, embedding_path
        return model

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

    def del_LI(self, W:np.array, b_w:np.array, U:np.array, b_u:np.array, X:scipy.sparse.csc_matrix, idx)->np.array:
        non_zero_cols = X[idx, :].nonzero()[1]
        Xi, Ji = X[idx, non_zero_cols].toarray()[0], non_zero_cols
        diff = U[Ji, :] @ W[idx, :] + b_u[Ji] + b_w[idx] - np.log(Xi)
        return ((2 * np.vectorize(self.f)(Xi) * diff).reshape(-1, 1).T @ U[Ji, :])[0]

    def del_sq_LI(self, U:np.array, X:scipy.sparse.csc_matrix, idx)->np.array:
        non_zero_cols = X[idx, :].nonzero()[1]
        Xi, Ji = X[idx, non_zero_cols].toarray()[0], non_zero_cols
        d = U.shape[1]
        temp = np.empty((Xi.shape[0], d))
        # couldn't figure out alternative of view()
        for i in range(d):
            temp[:, i] = np.sqrt(2. * np.vectorize(self.f)(Xi)) * U[Ji, i]
        return temp.T @ temp

    def f(self, x:np.float64, mx:np.float64=100., alpha:np.float64=0.75) -> np.float64:
        if x > mx:
            return np.float64(1.)
        return np.float64((x / mx) ** alpha)

    def compute_IF_deltas(self, document: str, M:GloveWrapper, X:scipy.sparse.csc_matrix):
        Y = self.doc2cooc(document, M.vocab, M.d)
        affected_inds = np.unique(Y.nonzero()[0])
        N = len(affected_inds)
        deltas = {}
        if N != 0:
            # TODO(ashutiwa): make this more efficient here https://seanlaw.github.io/2019/02/27/set-values-in-sparse-matrix/
            X_bar = X - Y
            X_bar[X_bar < 0] = 0
            X_bar.eliminate_zeros()
            for idx in affected_inds:
                gi = self.del_LI(M.W, M.b_w, M.U, M.b_u, X, idx)
                Hi = self.del_sq_LI(M.U, X_bar, idx)
                g_bar_i = self.del_LI(M.W, M.b_w, M.U, M.b_u, X_bar, idx)
                deltas[idx] = np.linalg.inv(Hi) @ (gi - g_bar_i)
        return deltas

    def inv_hessians_for(self, target_indices:np.array, M:GloveWrapper, X:scipy.sparse.csc_matrix):
        num_words = len(target_indices)
        H = {}
        for idx in target_indices:
            H[idx] = np.linalg.inv(self.del_sq_LI(M.U, X, idx))
        return H

    def gradients_for(self, target_indices:np.array, M:GloveWrapper, X:scipy.sparse.csc_matrix):
        num_words = len(target_indices)
        G = {}
        for idx in target_indices:
            G[idx] = self.del_LI(M.W, M.b_w, M.U, M.b_u, X, idx)
        return G

    def compute_IF_deltas_for(self, document, M:GloveWrapper, X:scipy.sparse.csc_matrix, target_indices:np.array, H:dict, G:dict):
        num_words = len(target_indices)
        deltas = {}
        if num_words != 0:
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


