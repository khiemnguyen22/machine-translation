import string
import re
from pickle import dump
from unicodedata import normalize
import argparse

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, mode='rt', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text
 
# split a loaded document into sentences
def to_sentences(doc, total_size):
	return doc.strip().split('\n')[:total_size]
 
# clean a list of lines
def clean_lines(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(table) for word in line]
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		line = ' '.join(line)
		line = '<start> ' + line + ' <end>'
		# store as string
		cleaned.append(line)
	return cleaned
 
# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
	dump(sentences, open(filename, 'wb'))
	print('Saved: %s' % filename)

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

    parser.add_argument('--total_size', type=int, default=1000)
    main(parser.parse_args())
