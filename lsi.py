import sys
# sys.path.append('libsvm/python/')
# sys.path.append('libsvm')
# from svmutil import *
import numpy
import scipy.sparse
import scipy.sparse.linalg
import numpy.linalg
import re
from collections import defaultdict
from random import shuffle
from tfidfSparse import tfidf
# from classes import get_classes

def get_numfeats(data):
	res = -1
	for e in data:
		for k in e:
			res = max(res, k)
	return res + 1

def lsi(data, ndims):
	'''points = numpy.zeros((len(data), get_numfeats(data)))
	for i, rep in enumerate(data):
			for feat in rep:
					points[i, feat] = rep[feat]

			l2norm = numpy.linalg.norm(points[i, :])
			if l2norm>0:
			    points[i, :]/=l2norm
	smat = scipy.sparse.csc_matrix(points)'''
	U, s, Vh = scipy.sparse.linalg.svds(smat, k=ndims)
	sigmatrix = numpy.matrix(numpy.zeros((ndims, ndims)))
	for i in range(ndims):
			sigmatrix[i, i] = s[i]
	return numpy.array(U[:, 0:ndims] * sigmatrix).tolist()

def main():
	features = lsi(tfidf('small_train.txt'), 100)
	classes = get_classes('small_train.txt', 0.7, 5)
	prob  = svm_problem(classes, features)
	param = svm_parameter('-t 0 -c 4 -b 1')
	m = svm_train(prob, param)
	test_features = lsi(tfidf('small_test.txt'), 100)
	test_classes = get_classes('small_test.txt', 0.7, 5)
	p_label, p_acc, p_val = svm_predict(test_classes, test_features, m)

if __name__ == '__main__':
  main()
