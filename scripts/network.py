"""
Module to train and use neural networks.
"""
import os
import keras
from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19
import scipy.misc
import numpy as np

PRETRAINED_MODELS = {
    "vgg16":        VGG16,
    "vgg19":        VGG19,
    "inception":    InceptionV3,
    "xception":     Xception,   #Only available for tensorflow
    "resnet":       ResNet50
}

class Network(object):
    
    def __init__(self, pretrained_arch, weights_path = None):
        """
        Transfer Learning network initialization.
        
        :param weights_path:
        :param pretrained_arch:
        """
        model = PRETRAINED_MODELS[pretrained_arch](weights='imagenet', include_top=False)
        
        if not weights_path:
            print 'Original Imagenet weights for network',pretrained_arch,'loaded'
        else:
            print 'Loading weights for',pretrained_arch,'from',weights_path
            #TODO
            #model.load_weights(weights)
        x = model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        predictions = keras.layers.Dense(7, activation='softmax')(x)
        self.pretrained_arch = pretrained_arch
        self.model = keras.models.Model(input=model.input, output=predictions)
        #print self.model.summary()
        return
        

class Temp(object):
    '''
    Temporal auxiliar class for debugging. Contains dummy versions of modules of the project that haven't been written yet.
    '''
    def __init__(self):
        return
    
    def get_dummy_generators(self):
        '''
        Returns a train and validation generators. The generators apply some very simple data augmentation (that has not been thoroughly tested at all). Each generator reads images from a different directory (I manually and randomly made a train validation split)
        '''
        train_dir = os.path.join('..','data','raw','train') #Should contain one director per class
        val_dir = os.path.join('..','data','raw','val') #Should contain one director per class
        from keras.preprocessing.image import ImageDataGenerator
        train_augmenter = ImageDataGenerator(
                rescale = 1./255.,
                shear_range=0.1,
                zoom_range=0.2,
                rotation_range=30,
                width_shift_range=0.1,
                height_shift_range=0.1,
                preprocessing_function=None, #Maybe we could place here our segmentation function
                horizontal_flip=True)

        val_augmenter = ImageDataGenerator(
                rescale = 1./255.,
                zoom_range=0.2)

        train_generator = train_augmenter.flow_from_directory(
                directory = train_dir,
                target_size=(150, 150),
                batch_size=32)

        val_generator = val_augmenter.flow_from_directory(
                directory = val_dir,
                target_size=(150, 150),
                batch_size=32)
        return train_generator, val_generator
    
    
    
    