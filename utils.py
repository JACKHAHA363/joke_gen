import json

def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
        word2idx = data
        idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word
