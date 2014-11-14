import sys
sys.path.append('../external_software/postagger/')
import numpy

import svm
import svr
import metrics

import tfidf
import bag_of_words
import sentiment
import pos

def kim_pos(filename):
	postags = pos.get_pos(filename)
	features = []

	# Used in Kim et al (2.1)
	# index 0: verbs
	# index 1: nouns
	# index 2: adj + adv
	# index 3: first perswon verbs
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
	return numpy.array(features)

def zhang_pos(filename):
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

	return numpy.array(features)

def kim_tfidf_ngrams(filename):
	return uni_features, bi_features

def many_sentiment(filename):
	return sentiment.get_sentiment_counts(filename)

if __name__ == "__main__":
	train_file = '/home/ak/Courses/cs73/project/dataset/small_train.txt'
	kim = kim_pos(train_file) # 5 features
	zhang = zhang_pos(train_file) # 7 features
	sent = many_sentiment(train_file) # 2 features

	X_train = numpy.hstack((kim, zhang, sent))
	t_train = svm.compile_targets(train_file)

	print X_train.shape

	model = svm.train(X_train, t_train)

	test_file = '/home/ak/Courses/cs73/project/dataset/small_test.txt'
	kim = kim_pos(test_file) # 5 features
	zhang = zhang_pos(test_file) # 7 features
	sent = many_sentiment(test_file) # 2 features

	X_test = numpy.hstack((kim, zhang, sent))
	t_test = svm.compile_targets(test_file)

	y_pred = svm.test(model, X_test)
	print y_pred.shape, numpy.sum(y_pred)
	metrics.run_classification_metrics(t_test, y_pred)
