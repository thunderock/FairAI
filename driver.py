# @Filename:    driver.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        5/28/22 10:19 PM


from models.word2vec import Word2Vec
from nltk import word_tokenize
from tqdm import tqdm
from gensim.corpora.wikicorpus import WikiCorpus
from utils.weat import WEAT
# ! wget -P /tmp/ http://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
# ! wget -P /tmp/ https://dumps.wikimedia.org/swwiki/latest/swwiki-latest-pages-articles.xml.bz2

# load file
# file = open("../enwik9.txt", "r")
# lines = file.readlines()
# sents = [word_tokenize(line.lower()) for line in tqdm(lines)]

# train the model
# file = '/tmp/swwiki-latest-pages-articles.xml.bz2'
# sents = list(WikiCorpus(file).get_texts())
# model = Word2Vec(sents)
# model.fit()
# model.save("../word2vec.model")


model = Word2Vec()
model.load("../word2vec.model")

weat = WEAT(model, 'weat/weat.json')

weat_scores = weat.scores
print(weat_scores)




