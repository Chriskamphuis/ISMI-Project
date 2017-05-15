import numpy as np
import random
from skimage.util import random_noise
from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate as rot
import time
import keras.preprocessing.image
    
class fliplr(object):
    
    def __init__(self):
        self.flip = False
    
    def augment(self, image, label=None):
        if label is not None:
            if self.flip:
                return np.fliplr(image), np.fliplr(label)
        if self.flip:
            return np.fliplr(image), None
        return image, None
        
    
    def randomize(self):
        self.flip = random.randint(0,1) == 1
                                  
class gauss_noise(object):
    
    def __init__(self):
        pass
    
    def augment(self, image, label=None):
        return random_noise(image), label
    
    def randomize(self):
        pass
    
class rotator(object):
    
    def __init__(self, max_angle):
        self.max_angle = max_angle
        self.directions = [-1, 1]
        self.angle = None
    
    def augment(self, image, label=None):
        image = rot(image, angle=self.angle, cval=1.0)
        if label is not None:
            rot(label, angle=self.angle, cval=1.0)
        return image, label
    
    def randomize(self):
        self.angle = np.random.uniform(-self.max_angle, self.max_angle)
        self.angle *= np.random.choice(self.directions)
    
class blur(object):
    
    def __init__(self, interval):
        self.interval = interval
        self.sigma = None
    
    def augment(self, image, label=None):
        return gaussian_filter(image, (0.0, self.sigma , self.sigma)), label
    
    def randomize(self):
        self.sigma = np.random.uniform(-self.interval, self.interval)
                                  
class chain_augmenters(object):
    
    def __init__(self, flip=True, noise=True, smooth=True,
                 blur_interval=0.3, rotate=True,
                 max_angle=5):
        self.augmenters = []
        if flip:
            self.augmenters.append(fliplr())
        if smooth:
            self.augmenters.append(blur(blur_interval))
        if noise:
            self.augmenters.append(gauss_noise())
        if rotate: # Takes about .4 seconds ..
            self.augmenters.append(rotator(max_angle))
        
    def augment(self, image, label=None):
        for aug in self.augmenters:
            image, label = aug.augment(image, label)
        return image, label
    
    def randomize(self):
        for aug in self.augmenters:
            aug.randomize()
            
class gamma_correct(object):
    
    def __init__(self):
        self.gamma = 2.0
    
    def augment(self, image, label=None):
        return exposure.adjust_gamma(image, self.gamma)
        
    def randomize(self):
        pass
    
    
class elastic_transform:
    
    def __init__(self):
        self.
    
    def augment(self, image, alpha, sigma, random_state=None):

        if random_state is None:
            random_state = numpy.random.RandomState(None)
 
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
 
        x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
        indices = numpy.reshape(y+dy, (-1, 1)), numpy.reshape(x+dx, (-1, 1))
 
        return map_coordinates(image, indices, order=1).reshape(shape)
    
if __name__ == "__main__":
    '''
    This code is to test if augmentation work
    Will remove it after I wrote all the augmentations
    '''
    aug = chain_augmenters(rotate=False)
    aug.randomize()
    test_image = '../data/testimage/frog.jpg'
    frog = imread(test_image)
    start = time.time()
    frog = aug.augment(frog)
    end = time.time()
    print(end - start)
    plt.imshow(frog[0])