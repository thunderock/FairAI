# @Filename:    word2vec_sampler.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        7/8/22 3:20 PM
import numpy as np
import faiss
import gravlearn
from scipy import sparse
from numba import njit

@njit(nogil=True)
def _csr_row_cumsum(indptr, data):
    out = np.empty_like(data)
    for i in range(len(indptr) - 1):
        acc = 0
        for j in range(indptr[i], indptr[i + 1]):
            acc += data[j]
            out[j] = acc
        # need to check this
    return out

@njit(nogil=True)
def _sample_one_neighbor(node_id, indptr, indices, data):
    neighbors = indices[indptr[node_id]:indptr[node_id + 1]]
    neighbors_weights = data[indptr[node_id]:indptr[node_id + 1]]
    return neighbors[np.searchsorted(neighbors_weights, np.random.rand())]


class Word2VecSampler(gravlearn.DataSampler):

    def __init__(self, in_vec, out_vec, alpha=.9, m=500, gpu_id=None):
        self.alpha = alpha
        self.in_vec = in_vec.astype(np.float32)
        self.out_vec = out_vec.astype(np.float32)
        self.center_sampler = gravlearn.FrequencyBasedSampler()
        self.n_elements, self.dim = out_vec.shape[0], self.out_vec.shape[1]
        n_train_sample = np.minimum(100000, self.n_elements)
        nlist = int(np.ceil(np.sqrt(n_train_sample)))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(self.dim), self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
        if gpu_id is not None:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, gpu_id, index)
        index.train(self.out_vec[np.random.choice(self.n_elements, n_train_sample, replace=False)])
        index.add(self.out_vec)

        # construct a graph of words with edges between a center word i and the m nodes with the largest $\exp(u_i ^\top v_j)$.
        dist, indices = index.search(self.in_vec, m)
        rows = np.arange(self.n_elements).reshape((-1, 1)) @ np.ones((1, m))
        rows, indices, dist = rows.ravel(), indices.ravel(), dist.ravel()
        s = indices >= 0
        rows, indices, dist = rows[s], indices[s], dist[s]
        dist = np.exp(dist)
        A = sparse.csr_matrix((dist, (rows, indices)), shape=(self.n_elements, self.n_elements))

        # preprocess the graph for faster sampling
        data = A.data / A.sum(axis=1).A1.repeat(np.diff(A.indptr))
        A.data = _csr_row_cumsum(A.indptr, data)
        self.A = A

    def fit(self, seqs): self.center_sampler.fit(seqs)

    def conditional_sampling(self, conditioned_on=None):
        if np.random.rand() < self.alpha:
            return _sample_one_neighbor(conditioned_on, self.A.indptr, self.A.indices, self.A.data)
        else:
            return self.center_sampler.sampling()[0]

    def sampling(self):
        cent = self.center_sampler.sampling()[0]
        cont = self.conditional_sampling(cent)
        return cent, cont

