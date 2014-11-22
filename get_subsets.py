import sys
import random

class LineIter:
	def __init__(self, filename, limit):
		self.file = open(filename, 'r')
		self.limit = limit
		self.count = 0
		
	def __iter__(self):
		return self

	def next(self):
		# Read one entry
		line = self.file.readline()

		if line == '' or self.count > self.limit:
			self.file.close()
			raise StopIteration()
		else:
			rating_num = 0
			rating = None
			text = None
			while rating_num < 10:
				rating = line
				rating_num = int(rating.split(':')[1].split('/')[1])
				text = self.file.readline()
				self.file.readline()
				if rating_num < 10:
					line = self.file.readline()
			self.count += 1
			return rating + text + '\n'


def write_subset(iterator, items_wanted, total_items):
	selected_items = []
	count = 0
	for item in iterator:
		if random.random() < (items_wanted + 0.0) / total_items:
				selected_items.append(item)
		count += 1
		if count % 10000 == 0:
			print count
	return selected_items

if __name__ == "__main__":
	train = write_subset(LineIter('train.txt', 1000000), 15000, 1000000)
	test = write_subset(LineIter('test.txt', 1000000), 10000, 1000000)

	with open('small_train.txt', 'w') as f:
		for str in train:
			f.write(str)

	with open('small_test.txt', 'w') as f:
		for str in test:
			f.write(str)



