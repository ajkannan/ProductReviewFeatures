import numpy
import matplotlib.pyplot as plt

kim = [[186, 326], [77, 411]]
mahony = [[205, 307], [123, 365]]
liu = [[231, 281], [137, 351]]
zhang = [[110, 402], [102, 386]]

sentiment = [[104, 408],[44, 444]]
tfidf = [[82, 430], [21, 467]]
pos = [[201, 311], [121, 367]]

kimzhang = [[204, 308], [91, 397]]
all = [[268, 217], [203, 302]]

data_1 = [kim, mahony, liu, zhang]
data_2 = [sentiment, tfidf, pos, all]
data = [kim, mahony, liu, zhang, sentiment, tfidf, pos, kimzhang, all]

# Show confusion matrix in a separate window

fig, axes = plt.subplots(nrows=2, ncols=2)
count = 0
titles_1 = ['Kim', "O'Mahony", 'Liu', 'Zhang']
for dat, ax in zip(data_1, axes.flat):
	im = ax.matshow(dat)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
	ax.set_title(titles_1[count])
	count += 1
	ax.set_xlabel('Predicted Label')
	ax.set_ylabel('True Label')
	
#plt.ylabel('True label')
#plt.xlabel('Predicted label')

cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
fig.tight_layout()
plt.show()







count = 0
titles_2 = ['Sentiment', 'tfidf', 'POS', 'All']
fig, axes = plt.subplots(nrows=2, ncols=2)
for dat, ax in zip(data_2, axes.flat):
	im = ax.matshow(dat)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
	ax.set_title(titles_2[count])
	ax.set_xlabel('Predicted Label')
	ax.set_ylabel('True Label')

	count += 1

plt.ylabel('True label')
plt.xlabel('Predicted label')

cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
fig.colorbar(im, cax=cax)
fig.tight_layout()
plt.show()
