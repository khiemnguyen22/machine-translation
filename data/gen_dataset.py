from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pickle
import random
import numpy as np

def load_datasets(filename):
    return np.array(pickle.load(open(filename, 'rb')))

def produce_train_test(en, fr, ratio = 0.9, total_size = 100):
    n_train = int(total_size * ratio)
    indexes = random.sample(range(total_size), n_train)
    trainX =  en[indexes]
    trainY = fr[indexes]

    testX = en[[i for i in range(total_size) if i not in indexes]]
    testY = fr[[i for i in range(total_size) if i not in indexes]]
    return trainX, trainY, testX, testY

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer(filters = '')
	tokenizer.fit_on_texts(lines)
	return tokenizer


# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X