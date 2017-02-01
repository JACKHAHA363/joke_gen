import pickle
import numpy as np
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Embedding, Input
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense
from keras.layers import TimeDistributed
import keras.backend as K
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *
import tensorflow as tf

# Read data
data = pickle.load(open("data/char_data.dat", "rb"))
idx_char = pickle.load(open("data/idx_char.dat","rb"))
char_idx = pickle.load(open("data/char_idx.dat","rb"))

# Build batched dataset
init_sentence = "you are too young and sometimes naive, "
window = len(init_sentence)
X = []
Y = [] 
max_len = len(data[0])
for s in data:
    for start in range(max_len - window):
        X.append(s[start : start + window])
        Y.append(s[start + window])

print("data done")
(_, num_chars) = np.shape(X[0])

# Build Model
model = Sequential()
model.add(LSTM(output_dim=256, input_shape=(window, num_chars), return_sequences=True))
model.add(LSTM(output_dim=128, return_sequences=False))
model.add(Dense(num_chars))
model.add(Activation('softmax'))
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.001))

print("model done")
# Generating 
def generate(temperature, init_sentence):
    print("start generating")
    generated = init_sentence
    
    # when generated is not long enough
    while len(generated) < 200:
        # take context and change into model input
        context = np.zeros([window, num_chars])
        for t, c in enumerate(generated[len(generated)-window : len(generated)]):
            context[t, char_idx[c]] = 1
        
        # call the model
        probs = model.predict(context.reshape([-1, window, num_chars]))[0]

        # sample
        tmp = np.log(probs) / temperature
        new_probs = np.exp(tmp) / np.sum(np.exp(tmp))
        next_idx = np.random.choice(num_chars, p=new_probs)
        
        generated += idx_char[next_idx]
    return generated

nb_epoch = 30
loss = []
val_loss = []

print("start training")
for e in range(nb_epoch):
    hist = model.fit(
            x=np.array(X), 
            y=np.array(Y), 
            batch_size=512,
            nb_epoch=1,
            verbose=1,
            validation_split=0.2,
            shuffle=True,
            )
    loss += hist.history['loss']
    val_loss += hist.history['val_loss']

    # generate sentences
    print(generate(0.35, init_sentence))

plt.figure()
plt.title("loss")
plt.plot(loss)
plt.savefig("result/training_loss")

plt.figure()
plt.title("val loss")
plt.plot(val_loss)
plt.savefig("result/validation_loss")

model.save("result/joke_gen.h5")
