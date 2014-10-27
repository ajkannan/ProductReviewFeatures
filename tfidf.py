import numpy as np
import math
import svr

def tfidf(train_file):
	idf = {} # word: number of documents with word in it

	curr_review_words = set()
	num_docs = 0
	with open(train_file, 'r') as f:
		line = f.readline()
		while line != '':
			# Iterate through file collecting counts for the entire corpus
			if 'review/text: ' in line:
				line = line[len('review/text: '):]
				words = line.split()
				for word in words:
					if word in idf:
						if word not in curr_review_words:
							idf[word] += 1
							curr_review_words.add(word)
					else:
						idf[word] = 1
						curr_review_words.add(word)
				num_docs += 1
				curr_review_words = set()
			line = f.readline()

	for word in idf:
		idf[word] = math.log(num_docs / idf[word])		

	word_index_lookup = {}
	keys = idf.keys()
	for i in xrange(len(keys)):
		word_index_lookup[keys[i]] = i
	tfidf_features = np.zeros((num_docs, len(idf)))

	# Iterate through the reviews to get frequency counts
	doc_i = 0
	with open(train_file, 'r') as f:
		line = f.readline()
		while line != '':
			if 'review/text: ' in line:
				line = line[len('review/text: '):]
				words = line.split()
				for word in words:
					tfidf_features[doc_i, word_index_lookup[word]] += (idf[word] / len(words))
				doc_i += 1
			line = f.readline()

	return tfidf_features

if __name__ == "__main__":
	''' For unit testing purposes only. ''' 
	features = tfidf('dataset/small_train.txt')

