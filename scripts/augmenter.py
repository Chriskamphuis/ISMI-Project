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
import keras
import cv2
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
import PIL
from PIL import Image

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
        dx = gaussian_filter((label.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
        dy = gaussian_filter((label.rand(*shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
 
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
 
 
class KAugmentor(keras.preprocessing.image.ImageDataGenerator):
    """
    Use Keras' Image Data Generator to augment an image
    """

    def __init__(self):
        self.imageDataGenerator = ImageDataGenerator(
                rescale = None,
                rotation_range = 70,
                zoom_range = [0.8, 1.2],
                width_shift_range = 0.12,
                height_shift_range = 0.12,
                horizontal_flip = True,
                shear_range = 0.15,
                channel_shift_range = 35.
                ) 
        
    def random_blur(self, x):
        """
        Add random blur asss well (not implemented in Keras)
        """
        radius = np.random.uniform(0, 2)
        x = PIL.Image.fromarray(x.astype("uint8"))
        x = x.filter(PIL.ImageFilter.GaussianBlur(radius=radius))
        x = np.array(x).astype("float32")
        return x
        
    def random_padding(self,old_im): 
        '''
        Receives a non-squared PIL Image object and returns
        its black-padded version as a Numpy array with square shape.
        the way the pad is added is random (to serve as augmentation),
        so this function is meant to be used online
        '''
        #old_im = Image.open(img_path)
        old_size = old_im.size

        new_size = (max(old_size),max(old_size))
        new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
        x = random.randint(0, int((new_size[0]-old_size[0])))
        y = random.randint(0, int((new_size[1]-old_size[1])))
        new_im.paste(old_im, (x,y))
        old_im = np.array(old_im)
        new_im = np.array(new_im)
        '''plt.subplot(1,2,1)
        plt.imshow(old_im)
        plt.title(str(old_im.shape))
        plt.subplot(1,2,2)
        plt.imshow(new_im)
        plt.title(str(new_im.shape))
        plt.show()'''
        return new_im
        
    def augment(self, x):
        """
        Augment one image
        """
        x = self.random_padding(x)
        x = self.random_blur(x)
        #print(x.mean())
        x = cv2.resize(x, dsize=(224,224))
        x = self.imageDataGenerator.random_transform(x.astype(K.floatx()))
        #print(x.mean())
        return x
        

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
