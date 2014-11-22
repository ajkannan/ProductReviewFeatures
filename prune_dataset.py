import os
import re
import random

rating_tag = 'review/helpfulness: '
text_tag = 'review/text: '
product_tag = 'product/productId: '
cats_tag = 'categories: '

def process(in_filename, opts, limit = float('inf')):
	in_file = open(in_filename, 'r')
	train_file = open('./dataset/train_with_categories.txt', 'a')
	test_file = open('./dataset/test_with_categories.txt', 'a')
	cats = get_categories()
	
	curr_rating = None
	curr_text = None
	curr_cats = None

	line = in_file.readline()
	count = 0
	while line != '' and count < limit:
		if line == '\n':
			# New review coming up, commit the current review
			# to file if it meets requirements
			num_ratings = int(curr_rating.split('/')[2])
			if num_ratings >= opts['min_ratings'] and num_ratings <= opts['max_ratings']:
				if random.random() < opts['train_ratio']:
					train_file.write(curr_cats + curr_rating + curr_text + '\n')
				else:
					test_file.write(curr_cats + curr_rating + curr_text + '\n')
					
		elif rating_tag in line:
			curr_rating = line
		elif text_tag in line:
			curr_text = line
		elif product_tag in line:
			curr_id = line.split(':')[1].strip()
			curr_cats = ' '.join(cats[curr_id]) + '\n'
			
		line = in_file.readline()
		count += 1
		if count % 5000000 == 0:
			print "Lines processed: ", count

	train_file.close()
	test_file.close()
	in_file.close()


def get_categories():
	r = re.compile(r'\w+')
	
	with open('./dataset/categories.txt', 'r') as f:
		last_product = None
		last_category = None
		line = f.readline()
		count = 0
		cats = {} # id : list of words
		
		while line != '':
			if line[0] == ' ':
				if last_product not in cats:
					cats[last_product] = []
				words = r.findall(line)
				for word in words:
					if word not in cats[last_product]:
						cats[last_product].append(word)
						
			else:
				last_product = line.strip()

			line = f.readline()
	 		count += 1

		return cats

if __name__ == "__main__":
	opts = {
		'min_ratings' : 4, 
		'max_ratings' : 30,
		'train_ratio' : 0.7
		}

	get_categories()
	process('./dataset/all.txt', opts)
