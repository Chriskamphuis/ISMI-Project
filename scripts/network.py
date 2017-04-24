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
        self.model = keras.models.Model(input=model.input, output=predictions)
        #print self.model.summary()
        return

class temp(object):
    '''
    Temporal auxiliar class for debugging. Contains dummy versions of modules of the project that haven't been written yet.
    '''
    def __init__(self):
        return
    
    def get_generators(self):
        '''
        Returns a train and validation generators.
        '''
        return
    
    
    
    