import numpy as np
import random
import matplotlib.pyplot as plt

def calc_stats(limit = float('inf')):
	with open('small_small_train.txt', 'r') as f:
		ratings = []
		length = []

		line = f.readline()
		count = 0
		curr_rating = None
		curr_text = None

		while line != '' and count < limit:
			if line == '\n':
				# New ratings
				rating_info = curr_rating.split(':')[1].split('/')
				pos_votes = int(rating_info[0])
				tot_votes = int(rating_info[1])
				usefulness = (pos_votes + 0.0) / tot_votes
				ratings.append(usefulness)

				text_pieces = curr_text.split(':')
				sum = 0
				for text in text_pieces[1:]:
					sum += len(text)
				length.append(sum)

				curr_rating = None
				curr_text = None
				count += 1
			elif curr_rating is None:
				curr_rating = line
			elif curr_text is None:
				curr_text = line
			else:
				print "error"

			line = f.readline()

		sample_indices = random.sample([i for i in xrange(count)], 1000)
		ratings_arr = np.array(ratings)
		lengths_arr = np.array(length)


		hist, bins = np.histogram(ratings, bins=10)
		width = 0.7 * (bins[1] - bins[0])
		center = (bins[:-1] + bins[1:]) / 2
		plt.bar(center, hist, align='center', width=width)
		plt.xlabel('Helpfulness Score')
		plt.ylabel('Number of Reviews')
		#plt.title('Histogram of Helpfulness Scores')
		plt.show()

		plt.scatter(ratings_arr[sample_indices], lengths_arr[sample_indices])
		plt.xlabel('Helpfulness Score')
		plt.ylabel('Review Length in Characters')
		#plt.title('Review Length vs Helpfulness Score')
		plt.xlim(-0.05,1.05)
		plt.ylim(-0.05,np.max(lengths_arr[sample_indices]) + 0.05)
		plt.show()

if __name__ == "__main__":
	calc_stats(1000)
