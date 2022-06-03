# @Filename:    dataset.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/3/22 12:38 PM

from gensim.corpora.wikicorpus import WikiCorpus

class Dataset(object):
    def __init__(self, path):
        self.path = path


    @property
    def lines(self):
        return list(WikiCorpus(self.path).get_texts())