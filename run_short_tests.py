import sys
sys.path.append('../external_software/postagger/')
import numpy
import re
import itertools
import scipy.sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
import matplotlib.pyplot as plt

import svm
import svr
import rfc
import rfr
import metrics
import tfidfSparse as ti
import bag_of_words
import sentiment
import pos
import lsi


def kim_pos(filename):
	postags = pos.get_pos(filename)
	features = []

	# Used in Kim et al (2.1)
	# index 0: verbs
	# index 1: nouns
	# index 2: adj + adv
	# index 3: first person verbs
	# index 4: verbs + adj + adv + nouns
	count = 0
	for r in postags:
		count += 1
		features.append([0.0] * 5)
		for t in r:
			if len(t) < 2:
				continue
			elif t[:2] == 'VB':
				features[-1][0] += r[t]
				features[-1][4] += r[t]
			elif t[:2] == 'NN':
				features[-1][1] += r[t]
				features[-1][4] += r[t]
			elif t[:2] == 'JJ':
				features[-1][2] += r[t]
				features[-1][4] += r[t]
			elif t[:2] == 'RB':
				features[-1][2] += r[t]
				features[-1][4] += r[t]
			elif len(t) > 2 and t[:3] == 'VBP':
				features[-1][3] += r[t]
				features[-1][4] += r[t]
	dense = numpy.array(features)
	sparse = scipy.sparse.csr_matrix(dense)
	return sparse

def zhang_features(filename):
	postags = pos.get_pos(filename)
	features = []
	
	# Index 0: Wh- words,
	# Index 1: proper nouns
	# Index 2: numbers
	# Index 3: modal verbs
	# Index 4: interjections
	# Index 5: comparative adjectives + adverbs
	# Index 6: superlative adjectivies + adverbs
	# Used in Zhang and Varadarajan (2.4)
	count = 0
	for r in postags:
		count += 1
		features.append([0.0] * 7)
		for t in r:
			if len(t) < 2:
				continue
			elif t[0] == 'W':
				features[-1][0] += r[t]
			elif t[:3] == 'NNP':
				features[-1][1] += r[t]
			elif t[:2] == 'CD':
				features[-1][2] += r[t]
			elif t[:2] == 'MD':
				features[-1][3] += r[t]
			elif t[:2] == 'UH':
				features[-1][4] += r[t]
			elif t[:3] == 'JJR' or t[:3] == 'RBR':
				features[-1][5] += r[t]
			elif t[:3] == 'JJS' or t[:3] == 'RBS':
				features[-1][6] += r[t]

	dense = numpy.array(features)
	sparse = scipy.sparse.csr_matrix(dense)
	return sparse

def tfidf_ngrams(train_filename, test_filename, with_lsi = True):
	unigrams_train = ti.tfidf(train_filename[:-4] + '_lemmatized.txt')
	unigrams_test = ti.tfidf(test_filename[:-4] + '_lemmatized.txt')
	keyIdMap = svm.getKeyIds(itertools.chain(unigrams_train, unigrams_test))

	X = svm.dictListToCSR(unigrams_train, keyIdMap = keyIdMap)
	Xp = svm.dictListToCSR(unigrams_test, keyIdMap = keyIdMap)

	svd = TruncatedSVD(n_components=100)
	X_trunc = svd.fit_transform(X)
	Xp_trunc = svd.transform(Xp)
	
	if with_lsi:
		return X_trunc, Xp_trunc
	else:
		return X, Xp

def kim_features(train_filename, test_filename):
	ngram_train, ngram_test = tfidf_ngrams(train_filename, test_filename, with_lsi = True)
	pos_train = kim_pos(train_filename)
	pos_test = kim_pos(test_filename)
	X_train = scipy.sparse.hstack((ngram_train, pos_train))
	X_test = scipy.sparse.hstack((ngram_test, pos_test))
	return X_train, X_test
	
def liu_features(filename):
	postags = pos.get_pos(filename)
	all_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
				'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR',
				'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP',
				'VBZ', 'WDT', 'WP',	'WP$', 'WRB', '-LRB-', '-RRB-', 'other']
	features = []
	for review in postags:
		features.append([0.0] * len(all_tags))
		for tag in review:
			try:
				features[-1][all_tags.index(tag)] = review[tag]
			except ValueError:
				features[-1][len(all_tags)-1] += review[tag]
	dense = numpy.array(features)
	sparse = scipy.sparse.csr_matrix(dense)
	return sparse
		
def many_sentiment(filename):
	dense = sentiment.get_sentiment_counts(filename)
	sparse = scipy.sparse.csr_matrix(dense)
	return sparse

def omahony_features(filename):
	# O'Mahony
	#     Review length in words
	#     Review length in characters
	#     Ratio of alphanumeric to non-alphanumeric
	#     Upper case to lower case
	features = []
	
	with open(filename, 'r') as f:
		line = f.readline()
		while line != '':
			# Iterate through file collecting counts for the entire corpus
			if 'review/text: ' in line:
				features.append([0.0] * 4)
				line = line[len('review/text: '):]
				words = line.split()
				features[-1][0] = len(words) + 0.0
				features[-1][1] = len(line) + 0.0

				# Count alphanumeric characters
				alpha_numeric = len(re.sub(r'\W+', '', line)) 
				features[-1][2] = alpha_numeric / (len(line) - alpha_numeric)

				# Count upper case characters
				uppercase_set = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
				uppercase = 0
				for c in line:
					if c in uppercase_set:
						uppercase += 1
				features[-1][3] = uppercase / (len(line) - uppercase)
						
			line = f.readline()

	dense = numpy.array(features)
	sparse = scipy.sparse.csr_matrix(dense)
	return sparse





def show_regression(pred, actual):
	plt.scatter(pred, actual)
	plt.show()

def main():
	train_file = '/home/ak/Courses/cs73/project/dataset/small_small_train.txt'
	test_file = '/home/ak/Courses/cs73/project/dataset/small_small_test.txt'

	sent_included = False
	train_feats = []
	test_feats = []
	if 'k' in sys.argv:
		kim_train, kim_test = kim_features(train_file, test_file)
		train_feats.append(kim_train)
		test_feats.append(kim_test)
		if not sent_included:
			train_feats.append(many_sentiment(train_file))
			test_feats.append(many_sentiment(test_file))
			sent_included = True
	if 'o' in sys.argv:
		train_feats.append(omahony_features(train_file))
		test_feats.append(omahony_features(test_file))
		if not sent_included:
			train_feats.append(many_sentiment(train_file))
			test_feats.append(many_sentiment(test_file))
			sent_included = True
	if 'l' in sys.argv:
		train_feats.append(liu_features(train_file))
		test_feats.append(liu_features(test_file))
		if not sent_included:
			train_feats.append(many_sentiment(train_file))
			test_feats.append(many_sentiment(test_file))
			sent_included = True
	if 'z' in sys.argv:
		train_feats.append(zhang_features(train_file))
		test_feats.append(zhang_features(test_file))
		sent_included = True
		if not sent_included:
			train_feats.append(many_sentiment(train_file))
			test_feats.append(many_sentiment(test_file))
			sent_included = True
	if 't' in sys.argv:
		tfidf_train, tfidf_test = tfidf_ngrams(train_file, test_file, with_lsi=False)
		train_feats.append(tfidf_train)
		test_feats.append(tfidf_test)

	X_train = None
	X_test = None
	if len(train_feats) > 1:
		X_train = scipy.sparse.hstack(train_feats)
		X_test = scipy.sparse.hstack(test_feats)
	else:
		X_train = train_feats[0]
		X_test = test_feats[0]

	svm.normalize(X_train)
	svm.normalize(X_test)

	# Classification
	# SV
	t_train_thresh = svm.compile_targets(train_file)
	t_test_thresh = svm.compile_targets(test_file)


	'''clf = ExtraTreesClassifier()
	X_new = clf.fit(X_train.toarray(), t_train_thresh).transform(X_train)
	for i in xrange(clf.feature_importances_.shape[0]):
		print i, clf.feature_importances_[i]'''

	'''bsvm = SVC(kernel="linear")
	selector = RFECV(bsvm, step=10)
	selector.fit(X_train, t_train_thresh)
	print selector.support_
	print selector.ranking_
	raw_input()'''

	class_model = None
	y_pred = None
	if 'rf' not in sys.argv:
		class_model = svm.train(X_train, t_train_thresh)
		y_pred = svm.test(class_model, X_test)
	else:
		class_model = rfc.train(X_train.todense(), t_train_thresh)
		y_pred = rfc.test(class_model, X_test.todense())
	metrics.run_classification_metrics(t_test_thresh, y_pred)
	print
	
	# Regression
	# SVR
	t_train = svr.compile_targets(train_file)
	t_test = svr.compile_targets(test_file)
	if 'rf' not in sys.argv:
		reg_model = svr.train(X_train, t_train)
		y_pred = svr.test(reg_model, X_test)
	else:
		reg_model = rfr.train(X_train.todense(), t_train)
		y_pred = rfr.test(reg_model, X_test.todense())

	#for i in xrange(X_test.shape[0]):
	#	print y_pred[i], t_train[i]
	metrics.run_regression_metrics(t_test, y_pred)
	
	show_regression(y_pred, t_test)


if __name__ == "__main__":
	main()
