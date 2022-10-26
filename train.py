import argparse
from data.gen_dataset import *

def tokenize(lang, sentences):
    tokenizer = create_tokenizer(sentences)
    vocab_size = len(tokenizer.word_index) + 1
    length = max_length(sentences)
    print(lang,' Vocabulary Size: %d' % vocab_size)
    print(lang,' Max Length: %d' % (length))
    return tokenizer, vocab_size, length

def main(args):
    print('Loading data...')
    in_sentences = load_datasets(args.input_data)
    out_sentences = load_datasets(args.target_data)
    print('input sentences size: ', in_sentences.shape)
    print('output sentences size: ', out_sentences.shape, '\n')

    print('Example sentence pairs:')
    in_lang, out_lang = args.input_data.split('.')[0], args.target_data.split('.')[0]
    print('Input ',in_lang, 'sentence: ', in_sentences[0])
    print('Target ',out_lang, 'sentence: ', out_sentences[0], '\n')

    trainX, trainY, testX, testY = produce_train_test(in_sentences, out_sentences)

    in_tokenizer, in_vocab_size, in_length = tokenize(in_lang, in_sentences)
    out_tokenizer, out_vocab_size, out_length = tokenize(out_lang, out_sentences)
    
    # prepare training data
    trainX = encode_sequences(in_tokenizer, in_length, trainX)
    trainY = encode_sequences(out_tokenizer, out_length, trainY)

    # prepare validation data
    testX = encode_sequences(in_tokenizer, out_length, testX)
    testY = encode_sequences(in_tokenizer, out_length, testY)


    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')

    parser.add_argument('--input_data', type=str, default='english.pkl')

    parser.add_argument('--target_data', type=str, default='french.pkl')

    main(parser.parse_args())
