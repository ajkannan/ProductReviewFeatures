import numpy as np
import math
import svr

def bag_of_words(filename):
	all_words = {} # word : index in the feature vector
	all_review_counts = [] # word: number of documents with word in it
	num_words = 0
	with open(filename, 'r') as f:
		line = f.readline()
		while line != '':
			# Iterate through file collecting counts for the entire corpus
			if 'review/text: ' in line:
				word_counts = {}
				line = line[len('review/text: '):]
				words = line.split()
				for word in words:
					if word in word_counts:
						word_counts[word] += 1.0
					else:
						word_counts[word] = 1.0
						all_words[word] = len(all_words)
				for word in word_counts:
					word_counts[word] /= len(words)

				all_review_counts.append(word_counts)
			line = f.readline()

	# Compile feature vector
	features = np.zeros((len(all_review_counts), len(all_words)))
	for i in xrange(len(all_review_counts)):
		curr_counts = all_review_counts[i]
		for word in curr_counts:
			features[i,all_words[word]] = curr_counts[word]

	return features

if __name__ == "__main__":
	features = bag_of_words('dataset/small_train.txt')
	print np.sum(features)
