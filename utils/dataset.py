# @Filename:    dataset.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/3/22 12:38 PM

from gensim.corpora.wikicorpus import WikiCorpus

class Dataset(object):
    def __init__(self, path, stream=False):
        self.path = path
        self.stream = stream

    @property
    def lines(self):
        if self.stream:
            return WikiCorpus(self.path).getstream()
        return list(WikiCorpus(self.path).get_texts())

    @property
    def size(self):
        return WikiCorpus(self.path).length


class TextCorpus(Dataset):
    def __init__(self, lines):
        super().__init__(path=None, stream=False)
        self._lines = lines

    @property
    def lines(self):
        return self._lines

    @property
    def size(self):
        return len(self._lines)
