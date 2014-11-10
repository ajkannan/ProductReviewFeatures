import numpy as np
import sklearn.metrics

''' This module contains functions for SVR regression as well as
    related utility methods (i.e. compiling the targets vector,
	normalizing features, and training/testing using support vector
	regression.  '''

def run_classification_metrics(correct, results):
	print "Accuracy"
	print sklearn.metrics.accuracy_score(correct, results)

	print "Precision"
	print sklearn.metrics.precision_score(correct, results)

	print "Recall"
	print sklearn.metrics.recall_score(correct, results)

	print "F1 Score"
	print sklearn.metrics.f1_score(correct, results)

	#print "ROC AUC"
	#print sklearn.metrics.roc_auc_score(correct, results)

	
def run_regression_metrics(correct, results):
	print "Mean Squared Error"
	print sklearn.metrics.mean_squared_error(correct, results)

	print "Explained variance"
	print sklearn.metrics.explained_variance_score(correct, results)

if __name__ == "__main__":
	''' For utility function testing purposes '''
	n_samples, n_features = 10, 5
	X = np.random.seed(0)
	y = np.random.randn(n_samples)
	X = np.random.randn(n_samples, n_features)
	svr = train(X, y)
	test(svr, X, y)
