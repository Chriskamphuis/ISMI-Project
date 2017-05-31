import numpy as np
import random
from skimage.util import random_noise
from scipy.ndimage import imread
from scipy.ndimage.filters import gaussian_filter
from skimage import exposure, transform
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate as rot
import time
from scipy.ndimage.interpolation import map_coordinates
    
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
    '''
    Maybe we should not use this because it really 
    makes it hard to classify an image
    '''
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
                                  
class gamma_correct(object):
    
    def __init__(self):
        self.gamma = None
    
    def augment(self, image, label=None):
        return exposure.adjust_gamma(image, self.gamma), label
        
    def randomize(self):
        self.gamma = np.random.uniform(.8, 1.2)
    
    
class elastic_transform(object):
    
    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma
    
    def augment(self, image, label=None):

        assert len(image.shape)==2
        
        if label is None:
            label = np.random.RandomState(None)
 
        shape = image.shape
        dx = gaussian_filter((label.rand(*shape) * 2 - 1),
                             self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((label.rand(*shape) * 2 - 1),
                             self.sigma, mode="constant", cval=0) * self.alpha
 
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
 
        return map_coordinates(image, indices, order=1).reshape(shape)
        
    def randomize(self):
        pass


class random_zooms(object):

    def __init__(self):
        self.zoom = np.random.uniform(1, 1.3)

    def augment(self, image, label=None):
        zoom_aug = transform.AffineTransform(scale = (1.0/self.zoom,
                                                      1.0/self.zoom))
        if label is None:
            return transform.warp(image, zoom_aug), label
        return transform.warp(image, zoom_aug), transform.warp(image, label)
        
    def randomize(self):
        self.zoom = np.random.uniform(1, 1.3)


class chain_augmenters(object):
    
    def __init__(self, flip=True, noise=True, smooth=True,
                 blur_interval=0.3, rotate=True,
                 max_angle=5, gamma=True, zoom=True, transform=True):
        self.augmenters = []
        if flip:
            self.augmenters.append(fliplr())
        if smooth:
            self.augmenters.append(blur(blur_interval))
        if noise:
            self.augmenters.append(gauss_noise())
        if rotate: # Takes about .4 seconds ..
            self.augmenters.append(rotator(max_angle))
        if gamma:
            self.augmenters.append(gamma_correct())
        if zoom:
            self.augmenters.append(random_zooms())
        if transform:
            pass
        
    def augment(self, image, label=None):
        for aug in self.augmenters:
            image, label = aug.augment(image, label)
        return image, label
    
    def randomize(self):
        for aug in self.augmenters:
            aug.randomize()
 

if __name__ == "__main__":
    '''
    This code is to test if augmentation work
    Will remove it after I wrote all the augmentations
    '''
    aug = chain_augmenters(rotate=False, noise=False)
    aug.randomize()
    test_image = '../data/images/pre/train/Type_1/1013.jpg'
    frog = imread(test_image)
    start = time.time()
    frog = aug.augment(frog)
    end = time.time()
    print(end - start)
    plt.imshow(frog[0])
