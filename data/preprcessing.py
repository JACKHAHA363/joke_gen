import numpy as np
import pickle
import json
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec

data = pickle.load(open("clean_data.dat", "rb"))
sentences = [simple_preprocess(x) for x in data]

model = Word2Vec(sentences)
weights = model.syn0
np.save(open("word_embedding.dat","wb"), weights)

vocab = dict([(k, v.index) for k, v in model.vocab.items()])
with open("vocab.dict", "w") as f:
    f.write(json.dumps(vocab))
