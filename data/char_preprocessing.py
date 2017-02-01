import numpy as np
import pickle

data = pickle.load(open("clean_data.dat", "rb"))
chars = set()
max_len = 0
for sentence in data:
    chars = chars.union(set(sentence.lower()))
    if len(sentence) > max_len:
        max_len = len(sentence)
stop_char = '@'
chars.add(stop_char)
print(len(chars))
char_idx = {ch:i for i, ch in enumerate(chars)}
idx_char = {i:ch for i, ch in enumerate(chars)}

def sentence2mat(sentence):
    result = np.zeros([max_len+1, len(chars)])
    for i in range(max_len+1):
        result[i][char_idx[stop_char]] = 1
    for i in range(len(sentence)):
        result[i][char_idx[sentence[i]]] = 1
        result[i][char_idx[stop_char]] = 0
    return result

new_data = [sentence2mat(x.lower()) for x in data]

pickle.dump(new_data, open("char_data.dat", "wb"))
pickle.dump(idx_char, open("idx_char.dat", "wb"))
pickle.dump(char_idx, open("char_idx.dat", "wb"))
