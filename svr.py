import numpy as np
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV as gs
from sklearn.preprocessing import normalize

''' This module contains functions for SVR regression as well as
    related utility methods (i.e. compiling the targets vector,
	normalizing features, and training/testing using support vector
	regression.  '''

def compile_targets(filename):
	# Construct target vector
	ratings = []
	with open(filename, 'r') as f:
		line = f.readline()
		while line != '':
			if 'review/helpfulness: ' in line:
				line = line[len('review/helpfulness: '):]
				numbers = line.split('/')
				percent = float(numbers[0]) / float(numbers[1])
				ratings.append(percent)
			line = f.readline()

	return np.array(ratings)

def normalize_features(X):
	np.normalize(X)

def train(X, y):
	svr = SVR()
	#svr = SVR(kernel='linear')
	svr.fit(X, y)
	return svr

def test(svr, X):
	return svr.predict(X)
	

if __name__ == "__main__":
	''' For utility function testing purposes '''
	n_samples, n_features = 10, 5
	X = np.random.seed(0)
	y = np.random.randn(n_samples)
	X = np.random.randn(n_samples, n_features)
	svr = train(X, y)
	test(svr, X, y)
