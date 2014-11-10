import numpy

def get_sentiment_dict():
	word_sent_dict = {}
	with open('../external_software/sentiment/swn.txt', 'r') as f:
		line = f.readline()
		while line != '':
			parts = line.split('\t')
			if '#' in parts[0]:
				line = f.readline()
				continue
			words = parts[4].split()
			pos = float(parts[2])
			neg = float(parts[3])
			for word in words:
				w = word.split('#')[0]
				if w not in word_sent_dict:
					word_sent_dict[w] = [0, 0, 0] # pos, neg, count
				prev_pos, prev_neg, prev_count = word_sent_dict[w]
				word_sent_dict[w][0] = (prev_pos * prev_count + pos) / (prev_count + 1)
				word_sent_dict[w][1] = (prev_neg * prev_count + neg) / (prev_count + 1)
				word_sent_dict[w][2] += 1
			line = f.readline()

	return word_sent_dict


def get_sentiment_counts(train_file):
	features = []
	sent_dict = get_sentiment_dict()
	curr_review_words = set()
	num_docs = 0
	not_found = 0
	found = 0
	with open(train_file, 'r') as f:
		line = f.readline()
		while line != '':
			# Iterate through file collecting counts for the entire corpus
			if 'review/text: ' in line:
				features.append([0.0] * 2) # positive, negative
				line = line[len('review/text: '):]
				words = line.split()
				for word in words:
					# Check if word in sentiment corpus
					if word in sent_dict:
						pos, neg, count = sent_dict[word]
						features[-1][0] += pos / len(words)
						features[-1][1] += neg / len(words)
						found += 1.0
					else:
						not_found += 1.0
			line = f.readline()

	return numpy.array(features)


if __name__ == "__main__":
	sd = get_sentiment_dict()
	print get_sentiment_counts('../dataset/small_train.txt')
