# @Filename:    dataset.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        6/3/22 12:38 PM

from gensim.corpora.wikicorpus import WikiCorpus

class Dataset(object):
    def __init__(self, path, stream=False):
        self.path = path
        self.stream = stream
        self.lines = list(WikiCorpus(self.path).get_texts()) if (path and not stream) else None

    # @property
    # def lines(self):
    #     if self.stream:
    #         return WikiCorpus(self.path).getstream()
    #     return self.__lines

    @property
    def size(self):
        return len(self.lines)


class TextCorpus(Dataset):
    def __init__(self, lines):
        super().__init__(path=None, stream=False)
        self.lines = lines
