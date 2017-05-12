'''
Module to train/finetune CNNs (using pretrained or custom architectures)
'''


import os
import keras
from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import scipy.misc
import numpy as np
from time import sleep
import segmentator



WEIGHTS_INPUT_DIR = os.path.join('..','data','weights','input')
WEIGHTS_OUTPUT_DIR = os.path.join('..','data','weights','output')
TENSORBOARD_LOGS_DIR = os.path.join('..','data','tensorboard_logs')

#TODO intengrate IMAGE_SHAPE in the code in a better way
IMAGE_SHAPE = (100, 100)



##################################
#PRETRAINED ARCHITECTURE AVAILABLE
##################################
PRETRAINED_ARCHS = {
    "vgg16":        VGG16,
    "vgg19":        VGG19,
    "inception":    InceptionV3,
    "xception":     Xception,   #Only available for tensorflow
    "resnet":       ResNet50
}

##################################
#CUSTOMARCHITECTURE AVAILABLE
##################################
'''
In order to try different custom architectures, just create a function
that returns a CNN model and add it to the CUSTOM_ARCHS dictionary (creating
a name/key for it as well), just as it is shown with "simple_cnn_1"
'''

def custom_1():
    model = Sequential()
    model.add(Conv2D(32, input_shape = IMAGE_SHAPE + (3,), kernel_size=(3, 3),activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model

CUSTOM_ARCHS ={
    "simple_cnn_1":    custom_1,
}



class Network(object):
    '''
    Class to train networks. The idea is not to train them from scratch, as the provided data set 
    was relatively small. Instead, we should employ transfer learning: first, we downloaded the weights 
    of pre-trained networks known to have good performance on ImageNet, with the aim of using it as a 
    reliable baseline feature extractor. Second, we substitute the last dense layer of the networks for
    a new one suitable for our case. Third, we used our data to train only this new layer (freezing the
    other layers). Finally, we unfreeze some of the original layers and train those with a low learning
    rate, to fine-tune the network for our specific domain.
    '''
    def __init__(self, arch, input_weights_name = None):
        '''
        Transfer Learning network initialization. There is no need to ask for the output_weights_name since those will be
        written to disk as training goes one and their name would be inferred from the name of the architecture used + the epoch
        + validation loss + valdiation accuracy
        
        :param input_weights_name: filepath of network weights to be read and loaded (must fit the selected
        architecture)
        :param arch: string defining the pretrained architecture to be used
        '''
        if arch in PRETRAINED_ARCHS:
            self.build_pretrained_arch(arch)
        else:
            self.model = CUSTOM_ARCHS[arch]()
       
        if not input_weights_name:
            print 'Original Imagenet weights for network',arch,'loaded'
        else:
            input_weights_path = os.path.join(WEIGHTS_INPUT_DIR, input_weights_name)
            print 'Loading weights for',arch,'from',input_weights_path
            self.model.load_weights(input_weights_path)
        #self.print_layers_info()
        self.generators = dict()
        self.arch = arch
        print self.model.summary()
        sleep(1)
        return
    
    def build_pretrained_arch(self, arch):
        pretrained_layers = PRETRAINED_ARCHS[arch](weights='imagenet', include_top=False)
        top_layers = pretrained_layers.output
        top_layers = keras.layers.GlobalAveragePooling2D()(top_layers)
        top_layers = keras.layers.Dense(1024, activation='relu')(top_layers)
        top_layers = keras.layers.Dense(3, activation='softmax')(top_layers)
   
        self.pretrained_layers = pretrained_layers
        self.model = keras.models.Model(
                                    inputs=self.pretrained_layers.input,
                                    outputs=top_layers)
    
    def print_layers_info(self):
        '''
        Prints information about current frozen (non trainable) and unfrozen (trainable)
        layers
        '''
        if self.arch in PRETRAINED_ARCHS:
            #Only pretrained archs have "pretrained_layers"
            print len(self.model.layers),'total layers (',len(self.pretrained_layers.layers),\
            'pretrained and',len(self.model.layers)-len(self.pretrained_layers.layers),'new stacked on top)'
        else:
            print len(self.model.layers),'total layers'
        trainable = [layer.trainable for layer in self.model.layers]
        non_trainable = [not i for i in trainable]
        tr_pos = list(np.where(trainable)[0])
        nontr_pos = list(np.where(non_trainable)[0])
        if len(nontr_pos) > 0:
            print '\t',sum(non_trainable),'non-trainable layers: from',nontr_pos[0],'to',nontr_pos[-1]
        print '\t',sum(trainable),'trainable layers: from',tr_pos[0],'to',tr_pos[-1]
        print 'Trainable layer map:',''.join([str(int(l.trainable)) for l in self.model.layers])
    
    def freeze_all_pretrained_layers(self):
        '''
        Freeze all the pretrained layers. Note: a "pretrained layer" is named as such
        even after fine-tunning it
        '''
        print 'Freezing all pretrained layers...'
        for layer in self.pretrained_layers.layers:
            layer.trainable = False
    
    def unfreeze_last_pretrained_layers(self, n_layers = None, percentage = None):
        '''
        Un freeze some of the last pretrained layers of the model
        '''
        assert n_layers or percentage
        if percentage:
            assert percentage < 1
            n_layers = int(float(len(self.pretrained_layers.layers))*percentage)
        print 'Freezing last',n_layers,'of the pretrained model',self.arch,'...'
        for layer in self.pretrained_layers.layers[-n_layers:]:
            layer.trainable = True
    
    def set_train_val_generators(self, train_generator, val_generator):
        '''
        Associate training and validation generators
        '''
        self.generators['train'] = train_generator
        self.generators['validate'] = val_generator
        return
    
    def compile(self, finetuning):
        '''
        Compile the model (required before training). In finetuning we
        apply a small learning rate to avoid messing up the pretrained
        weights. In other case we use the adam optimizer
        '''
        if finetuning:
            optimizer = keras.optimizers.SGD(lr=0.0001, momentum=0.9)
        else:
            optimizer = 'adam'
        print 'Using optimizer:',optimizer
        #Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics = ['accuracy'])
        return
        
    
    def train(self, epochs, batch_size):
        '''
        Train the network. Note: only the weights of the trainable layers will
        be modified
        '''
        
        #Check that the top layers are trainable
        #assert(all([layer.trainable for layer in self.top_layers.layers]))
        
        #Visualize which layers are gonna be trained
        self.print_layers_info()
        
        callbacks_list = []
        trainable_layers = sum([int(layer.trainable) for layer in self.model.layers])
        #Set up save checkpoints for model's weights
        weights_name = self.arch +'.'+str(trainable_layers) + ".e{epoch:03d}-tloss{loss:.4f}-vloss{val_loss:.4f}.hdf5"
        weights_path = os.path.join(WEIGHTS_OUTPUT_DIR, weights_name)
        callbacks_list.append(keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor = 'val_loss',
            verbose=1,
            save_best_only = True,
            mode = 'min'))
        
        #Set up tensorboard logs
        callbacks_list.append(keras.callbacks.TensorBoard(
                log_dir = TENSORBOARD_LOGS_DIR,
                histogram_freq = 1,
                write_graph = True,
                write_images = True))
        
        #TODO handle usage of class freq weights or balanced batch generators
        
        class_weights = {
                0:1-0.18,
                1:1-0.55,
                2:1-0.27}
        
        #class_weights = None
        
        #Fix needed https://github.com/fchollet/keras/issues/5475
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        # Train
        total_unique_images = 1481.
        self.model.fit_generator(
            generator = self.generators['train'],
            steps_per_epoch = int(0.75*total_unique_images/batch_size), 
            epochs = epochs,
            validation_data = self.generators['validate'],
            validation_steps = int(0.25*total_unique_images/batch_size),
            class_weight = class_weights,
            workers = 10,
            callbacks = callbacks_list)
        return
	
	def predict(self, X):
		return self.model.predict(X,batch_size=1)
class Temp(object):
    '''
    Temporal auxiliar class for debugging. Contains dummy versions of modules of the project
    that haven't been written yet.
    '''
    def __init__(self):
        return
    
    def get_dummy_generators(self):
        '''
        Returns a train and validation generators. The generators apply some very simple 
        data augmentation (that has not been thoroughly tested at all). Each generator reads 
        images from a different directory (I manually and randomly made a train validation 
        split)
        '''
        train_dir = os.path.join('..','data','images','raw','train') #Should contain one director per class
        val_dir = os.path.join('..','data','images','raw','validate') #Should contain one director per class
        
        
        
        from keras.preprocessing.image import ImageDataGenerator
        train_augmenter = ImageDataGenerator(
                rescale = 1./255.,
                shear_range=0.1,
                zoom_range=0.2,
                rotation_range=30,
                width_shift_range=0.15,
                height_shift_range=0.15,
                preprocessing_function=None, #Maybe we could place here our segmentation function, or add random blur
                horizontal_flip=True)

        val_augmenter = ImageDataGenerator(
                rescale = 1./255.,
                zoom_range=0.2)

        train_generator = train_augmenter.flow_from_directory(
                directory = train_dir,
                target_size=IMAGE_SHAPE,
                batch_size=32,
                classes = ['Type_1','Type_2','Type_3'])

        val_generator = val_augmenter.flow_from_directory(
                directory = val_dir,
                target_size=IMAGE_SHAPE,
                batch_size=32,
                classes = ['Type_1','Type_2','Type_3'])
                
        return train_generator, val_generator


