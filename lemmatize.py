from nltk.stem import WordNetLemmatizer
import string
import re
wnl = WordNetLemmatizer()

def write_lemmatize(filename):
	exclude = set(string.punctuation)

	with open(filename[:-4] + '_lemmatized.txt', 'w') as output:
		with open(filename, 'r') as input:
			line = input.readline()
			while line != '':
				words = line.split()
				for word in words:
					word_without_punc = re.sub(r'\W+', '', word).lower()
					lemmatized = wnl.lemmatize(word_without_punc)
					if word_without_punc != 'reviewhelpfulness' and word_without_punc != 'reviewtext':
						output.write(lemmatized + ' ')
					else:
						output.write(word + ' ')
				output.write('\n')
				line = input.readline()


if __name__ == "__main__":
	write_lemmatize('../dataset/small_train.txt')
	write_lemmatize('../dataset/small_test.txt')
