from __future__ import division
import sys
import math
import time
import re

# This code is in need of polishing: refactor
# if time is available - Will

# Class to classify non-native english speaker essays
# based off of biram models and resulting perplexity
class ClassifyEssays:

	def __init__(self, lamb, eps):
		
		# lambda parameter for smoothing
		self.lamb = lamb
		# episilon parameter for small probability just
		# greater than 0
		self.eps = eps
		# ngram model which contains dictionaries
		self.ngram = {}

	def buildNgram(self, filename, threshold = .6, num = 10):

		self.ngram["helpful"] = {}
		self.ngram["unhelpful"] = {}
			
		count = self.fileParse(filename, self.ngram, threshold, num)
		
		# Convert counts into probabilities
		for helpfulness in self.ngram.keys():
			self.convertToProb(count[helpfulness], self.ngram[helpfulness])

		# print len(self.ngram["helpful"])
		# print len(self.ngram["unhelpful"])
  
  	# Converts counts in a bigram dictionary to probabilities,
  	# then adds the probabilities of the unigrams to the dictionary
	def convertToProb(self, uniCounts, biCounts):
		for word in biCounts.keys():
			context = word.split("\t")[1]
			biCounts[word] *= 1.0 / uniCounts[context]

		totalUniCounts = sum(uniCounts.values())
		for context in uniCounts:
				uniCounts[context] *= 1.0 / totalUniCounts
				biCounts[context] = uniCounts[context]


	def fileParse(self, filename, dictionary, threshold, num):
		f = open(filename, 'r')
		count = {}
		count["helpful"] = {}
		count["unhelpful"] = {}

		key = None
		context = "<s>"
		for i, line in enumerate(f):
			lineList = line.strip().split()
			if len(lineList) < 1: continue
			if lineList[0] == "review/helpfulness:":
				context = "<s>"
				helpfulFraction = lineList[1].split("/")
				good = float(helpfulFraction[0])
				total = float(helpfulFraction[1])
				if good / total >= threshold and total >= num: # don't forget to change the other one in classify
					key = "helpful"
				else:
					key = "unhelpful"
			else:
				for unit in line.strip().split():
					if unit == "review/text:": continue
					unit = re.sub(r"[^a-z]", "", unit)
					if not context in count[key]:
						count[key][context] = 1
					else:
						count[key][context] += 1

					contextWord = unit.lower() + "\t" + context
					if contextWord not in dictionary[key]:
						dictionary[key][contextWord] = 1
					else:
						dictionary[key][contextWord] += 1

					context = unit.lower()
				if not context in count[key]:
					count[key][context] = 1
				else:
					count[key][context] += 1
		f.close()
		return count

	def classify(self, filename):
		f = open(filename, 'r')

		key = None

		correct = 0
		incorrect = 0

		probabilities = {}
		count = {}
		context = "<s>"
		perplexities = []
		bigrams = []
		actuals = []
		for i, line in enumerate(f):
			lineList = line.strip().split()
			if len(lineList) < 1: continue
			if lineList[0] == "review/helpfulness:":
				probabilities = {}
				count = {}
				context = "<s>"
				helpfulFraction = lineList[1].split("/")
				good = float(helpfulFraction[0])
				total = float(helpfulFraction[1])
				if good / total >= .6 and total >= 5: # don't forget to change the other one in fileParse
					key = "helpful"
				else:
					key = "unhelpful"
			else:
				for unit in line.strip().split():
					if unit == "review/text:":
						continue

					unit = re.sub(r"[^a-z]", "", unit)

					if not context in count:
						count[context] = 1
					else:
						count[context] += 1

					contextWord = unit.lower() + "\t" + context
					if contextWord not in probabilities:
						probabilities[contextWord] = 1
					else:
						probabilities[contextWord] += 1

					context = unit.lower()
				if not context in count:
					count[context] = 1
				else:
					count[context] += 1

				self.convertToProb(count, probabilities)

				minPerplexity = float("inf")
				minKey = "N/A"

				#print "Compare: "
				actuals.append(key)
				reviewPerp = []
				perplexities.append(reviewPerp)
				bigrams.append(probabilities)
				for proposedkey in self.ngram.keys():
					perplexity = self.calcPerplexity(self.ngram[proposedkey], \
					probabilities)
					reviewPerp.append(perplexity)
				#	print perplexity, proposedkey
					if perplexity < minPerplexity:
						minPerplexity = perplexity
						minKey = proposedkey

				#print "Actual: " + key
				# print minKey, key
				if minKey == key and minKey == "unhelpful":
					correct += 1
				elif minKey == "unhelpful":
					incorrect += 1
		f.close()
		# for i, vector in enumerate(perplexities):
		# 	print vector,
		# 	if vector[0] > vector[1]:
		# 		print "helpful",
		# 	else:
		# 		print "unhelpful",
		# 	print "Actual: " + actuals[i]

		print  correct / (correct + incorrect)

		return perplexities, bigrams

	def calcPerplexity(self, Q, P):
		# Cross Entropy is -sum(x in X, c in C) of
		# P(x, c) log Q(x|c)
		crossEntropy = 0
		for unit in P.keys():
			# skip unigrams that were added to dictionary
			if len(unit.split("\t")) == 1:
				continue
			# ignore context if it is <s>
			context = unit.split("\t")[1]
			if context == "<s>":
				unit = unit.split("\t")[0]

			# Actual calculation
			if unit in Q:
				Qunit = Q[unit]
			else:
				baseWord = unit.split("\t")[0]
				if baseWord in Q:
					QbaseWord = Q[baseWord]
				else:
					QbaseWord = self.eps

				Qunit = (1 - self.lamb) * QbaseWord

			crossEntropy -= P[unit] * P[context] *  math.log(Qunit, 2)

		return math.pow(2, crossEntropy)

def main():
	# Read commandline arguments
	a = time.clock()

	model = ClassifyEssays(lamb = .03, eps = 1e-6)
	model.buildNgram("random_train.txt", num = 5)
	perplexities, bigrams = model.classify("random_test.txt")

	b = time.clock() - a
	print "Time taken for bigram:" + str(b)


if __name__ == "__main__":
	main()