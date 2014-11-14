import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import normalize

''' This module contains functions for SVR regression as well as
    related utility methods (i.e. compiling the targets vector,
	normalizing features, and training/testing using support vector
	regression.  '''

def compile_targets(filename, threshold = 0.78):
	# Construct target vector
	ratings = []
	with open(filename, 'r') as f:
		line = f.readline()
		while line != '':
			if 'review/helpfulness: ' in line:
				line = line[len('review/helpfulness: '):]
				numbers = line.split('/')
				percent = float(numbers[0]) / float(numbers[1])
				if percent > threshold:
					ratings.append(1.0)
				else:
					ratings.append(0.0)
			line = f.readline()

	return np.array(ratings)

def normalize_features(X):
	np.normalize(X)

def train(X, y):
	svm = SVC()
	svm.fit(X, y)
	return svm

def test(svm, X):
	return svm.predict(X)

if __name__ == "__main__":
	''' For utility function testing purposes '''
	for thresh in (0.75, 0.77, 0.78):
		targs = compile_targets('../dataset/small_train.txt', thresh)
		print thresh, ':', (np.sum(targs) + 0.0) / targs.shape[0]

	n_samples, n_features = 10, 5
	X = np.random.seed(0)
	y = np.random.randn(n_samples)
	X = np.random.randn(n_samples, n_features)
	svm = train(X, y)
	test(svm, X)
