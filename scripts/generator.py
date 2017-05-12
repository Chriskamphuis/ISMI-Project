import os
import numpy as np


class BatchGenerator(object):
        """Generate batches of data"""

        def __init__(self):
            pass

        def generate(self, data, labels, batch_size=32,
                     balanced=True, shuffle=True):
            while True:
                if shuffle:
                    shuf(data, labels)
                if balanced:
                    nr_classes, counts = np.unique(labels, return_counts=True)
                    smallest = np.amin(counts)
                else:
                    batches = len(data/batch_size)
                    for batch in range(batches):
                        x = data[batch*batch_size:(batch+1)*batch_size,
                                 :, :, :]
                        y = labels[batch*batch_size:(batch+1)*batch_size]
                        yield((x, y))

        def shuf(data, labels):
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            return data[indices], labels[indices]
