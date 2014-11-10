import subprocess
import os

def write_pos_file(filename):
	with open(filename[:-4] + '_parsed.txt', 'w') as output:
		subprocess.call(['java', '-cp', "*", 'edu.stanford.nlp.tagger.maxent.MaxentTagger', '-model', 'models/english-left3words-distsim.tagger', '-textFile', filename, '-outputFormat', 'inlineXML', '-outputFormatOptions', 'lemmatize'], stdout=output)

def get_pos(filename):
	if not os.path.isfile(filename[:-4] + '_parsed.txt'):
		write_pos_file(filename)
		
	postag_unigrams = []
	postag_bigrams = []
	prev_pos = '<s>'
	curr_review_length = 0
	with open(filename[:-4] + '_parsed.txt', 'r') as f:
		line = f.readline()
		line = f.readline()
		line = f.readline()
		while line != '':
			if 'word wid=' in line:
				# Check if we're at the start of a new review.  If so, ignore the current line and start new list
				if 'review/helpfulness' in line:
					line = f.readline() # Get rid of the colon afterwards
					line = f.readline()
				elif 'review/text' in line:
					line = f.readline() # Get rid of the colon afterwards
					if len(postag_unigrams) > 0:
						for key in postag_unigrams[-1]:
							postag_unigrams[-1][key] /= curr_review_length
					postag_unigrams.append({})
					postag_bigrams.append({})
					curr_review_length = 0
					prev_pos = '<s>'
				else:
					curr_review_length += 1
					pos_start = line.index('" pos="') + len('" pos="')
					pos_end = line.index('"', pos_start + 1)
					pos = line[pos_start:pos_end]
					
					# Add to unigrams
					if pos not in postag_unigrams[-1]:
						postag_unigrams[-1][pos] = 0.0
					postag_unigrams[-1][pos] += 1.0

					# Add to bigrams
					if prev_pos not in postag_bigrams[-1]:
						postag_bigrams[-1][prev_pos] = {}
					if pos not in postag_bigrams[-1][prev_pos]:
						postag_bigrams[-1][prev_pos][pos] = 0.0
					postag_bigrams[-1][prev_pos][pos] += 1.0
					
					prev_pos = pos
			line = f.readline()

	# Normalize last review
	for key in postag_unigrams[-1]:
		postag_unigrams[-1][key] /= curr_review_length

	return postag_unigrams
	
if __name__ == "__main__":
	 postag_unigrams = get_pos("../../dataset/small_train.txt")
	 postag_unigrams = get_pos("../../dataset/small_test.txt")
	 print postag_unigrams
