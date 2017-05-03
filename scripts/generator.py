import os
import numpy as np

class BatchGenerator(object):
	"""Generate batches of data"""

	
	def __init__(self):
	
	
	def generate(self, data, labels, batch_size = 32, balanced=True, shuffle=True):
		while True:
			if shuffle:
				shuf(data, labels)
			if balanced:
				nr_classes, counts = np.unique(labels, return_counts=True)
				smallest = np.amin(counts)
				for label in nr_classes:
				#undersample? If so, remove all excess data from other classes and then make batches as below? Or sample batches that contain same amount of data from each class?				
			else:				
				batches = int(len(data/batch_size)
				for batch in range(batches):
					x = data[batch*batch_size:(batch+1)*batch_size, :, :, :]
					y = labels[batch*batch_size:(batch+1)*batch_size]
					yield((x,y))



	def shuf(data, labels):
		indices = np.arange(len(data))
		np.random.shuffle(indices)
		return data[indices], labels[indices]