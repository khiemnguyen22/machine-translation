from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pickle
import random
import numpy as np
from data.preprocessing import create_manythings_dataset

def load_datasets(filename):
    return np.array(pickle.load(open(filename, 'rb')))

def produce_train_test(en, fr, ratio = 0.9, total_size = 1000):
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
	X = tokenizer.texts_to_sequences(lines)
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

def tokenize(lang, sentences):
    tokenizer = create_tokenizer(sentences)
    vocab_size = len(tokenizer.word_index) + 1
    length = max_length(sentences)
    print(lang,' Vocabulary Size: %d' % vocab_size)
    print(lang,' Max Length: %d' % (length))
    return tokenizer, vocab_size, length

def gen_dataset(input_data, target_data, num_samples):
	print('Loading data...')
	in_sentences = load_datasets(input_data)
	out_sentences = load_datasets(target_data)
	print('input sentences size: ', in_sentences.shape)
	print('output sentences size: ', out_sentences.shape, '\n')

	print('Example sentence pairs:')
	in_lang, out_lang = input_data.split('.')[0], target_data.split('.')[0]
	print('Input ',in_lang, 'sentence: ', in_sentences[0])
	print('Target ',out_lang, 'sentence: ', out_sentences[0], '\n')

	trainX, trainY, testX, testY = produce_train_test(in_sentences, out_sentences, total_size = num_samples)

	in_tokenizer, in_vocab_size, in_length = tokenize(in_lang, in_sentences)
	out_tokenizer, out_vocab_size, out_length = tokenize(out_lang, out_sentences)

	# prepare training data
	trainX = encode_sequences(in_tokenizer, in_length, trainX)
	trainY = encode_sequences(out_tokenizer, out_length, trainY)

	print('[Training] input size: ', trainX.shape, ', target size: ', trainY.shape)

	# prepare validation data
	testX = encode_sequences(in_tokenizer, out_length, testX)
	testY = encode_sequences(in_tokenizer, out_length, testY)
	print('[Testing] input size: ', testX.shape, ', target size: ', testY.shape)
	return trainX, trainY, testX, testY, in_tokenizer, out_tokenizer, in_vocab_size, out_vocab_size

def gen_manythings_dataset(path, num_samples, in_lang, out_lang):
	in_sentences, out_sentences = create_manythings_dataset(path, None)

	trainX, trainY, testX, testY = produce_train_test(np.array(in_sentences[:num_samples]), np.array(out_sentences[:num_samples]), total_size= num_samples)
	
	in_tokenizer, in_vocab_size, in_length = tokenize(in_lang, in_sentences)
	out_tokenizer, out_vocab_size, out_length = tokenize(out_lang, out_sentences)

	# prepare training data
	trainX = encode_sequences(in_tokenizer, in_length, trainX)
	trainY = encode_sequences(out_tokenizer, out_length, trainY)

	print('[Training] input size: ', trainX.shape, ', target size: ', trainY.shape)

	# prepare validation data
	testX = encode_sequences(in_tokenizer, out_length, testX)
	testY = encode_sequences(in_tokenizer, out_length, testY)
	print('[Testing] input size: ', testX.shape, ', target size: ', testY.shape)
	return trainX, trainY, testX, testY, in_tokenizer, out_tokenizer, in_vocab_size, out_vocab_size

