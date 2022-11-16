import string
import re
from pickle import dump
from unicodedata import normalize
import argparse
import io

# load doc into memory
def load_doc(filename):

	file = open(filename, mode='rt', encoding='utf-8')

	text = file.read()

	file.close()
	return text
 
# split a loaded document into sentences
def to_sentences(doc, total_size):
	return doc.strip().split('\n')[:total_size]
 
def preprocess_line(line):

	re_print = re.compile('[^%s]' % re.escape(string.printable))

	table = str.maketrans('', '', string.punctuation)

	line = normalize('NFD', line).encode('ascii', 'ignore')
	line = line.decode('UTF-8')

	line = line.split()

	line = [word.lower() for word in line]

	line = [word.translate(table) for word in line]

	line = [re_print.sub('', w) for w in line]

	line = [word for word in line if word.isalpha()]
	line = ' '.join(line)
	line = '<start> ' + line + ' <end>'
	return line

# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	for line in lines:
		line = preprocess_line(line)
		cleaned.append(line)
	return cleaned
 
# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

def create_manythings_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

    word_pairs = [[preprocess_line(w) for w in l.split('\t', 2)[:-1]]  for l in lines[:num_examples]]
    return zip(*word_pairs)

def preprocessing_pipeline(file, out_name, total_size):
    doc = load_doc(file)
    sentences = to_sentences(doc, total_size)
    cleaned = clean_lines(sentences)
    save_clean_sentences(cleaned, out_name)

def main(args):
    print('Preprocessing...')
    preprocessing_pipeline(args.input_data, args.input_lang, args.total_size)
    preprocessing_pipeline(args.target_data, args.target_lang, args.total_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data preprocessing')

    parser.add_argument('--input_lang', type=str, default='english.pkl')

    parser.add_argument('--target_lang', type=str, default='french.pkl')

    parser.add_argument('--input_data', type=str, default='europarl-v7.fr-en.en')

    parser.add_argument('--target_data', type=str, default='europarl-v7.fr-en.fr')

    parser.add_argument('--total_size', type=int, default=10000)
    main(parser.parse_args())
