import os
import keras
from keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19
from keras.layers import Dense, Reshape, Flatten
from keras.models import Sequential
import scipy.misc
import numpy as np
from time import sleep

PRETRAINED_MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,  # Only available for tensorflow
    "resnet": ResNet50
}

WEIGHTS_INPUT_DIR = os.path.join('..', 'data', 'weights', 'input')
WEIGHTS_OUTPUT_DIR = os.path.join('..', 'data', 'weights', 'output')
TENSORBOARD_LOGS_DIR = os.path.join('..', 'data', 'tensorboard_logs')


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

    def __init__(self, pretrained_arch, input_weights_name=None, output_weights_name=None):
        '''
        Transfer Learning network initialization.

        :param input_weights_name:
        :param pretrained_arch:
        '''
        self.model = Sequential()
        self.model.add(Flatten(input_shape=[224,224,3]))
        self.model.add(Dense(20, name="dense_1"))  # will be loaded
        self.model.add(Dense(3, name="new_dense"))  # will not be


        # print self.model.summary()
        return

    def print_layers_info(self):
        '''
        Prints information about current frozen (non trainable) and unfrozen (trainable)
        layers
        '''
        print len(self.model.layers), 'total layers (', len(self.pretrained_layers.layers), \
            'pretrained and', len(self.model.layers) - len(self.pretrained_layers.layers), 'new stacked on top)'
        trainable = [layer.trainable for layer in self.model.layers]
        non_trainable = [not i for i in trainable]
        tr_pos = list(np.where(trainable)[0])
        nontr_pos = list(np.where(non_trainable)[0])
        print '\t', sum(non_trainable), 'non-trainable layers: from', nontr_pos[0], 'to', nontr_pos[-1]
        print '\t', sum(trainable), 'trainable layers: from', tr_pos[0], 'to', tr_pos[-1]
        print 'Trainable layer map:', ''.join([str(int(l.trainable)) for l in self.model.layers])

    def freeze_all_pretrained_layers(self):
        '''
        Freeze all the pretrained layers. Note: a "pretrained layer" is named as such
        even after fine-tunning it
        '''
        print 'Freezing all pretrained layers...'
        for layer in self.pretrained_layers.layers:
            layer.trainable = False

    def unfreeze_last_pretrained_layers(self, n_layers=None, percentage=None):
        '''
        Un freeze some of the last pretrained layers of the model
        '''
        assert n_layers or percentage
        if percentage:
            assert percentage < 1
            n_layers = int(float(len(self.pretrained_layers.layers)) * percentage)
        print 'Freezing last', n_layers, 'of the pretrained model', self.pretrained_arch, '...'
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
        print 'Using optimizer:', optimizer
        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=['accuracy'])
        self.model.summary()
        return

    def train(self, epochs, batch_size):
        '''
        Train the network. Note: only the weights of the trainable layers will
        be modified
        '''

        # Check that the top layers are trainable
        # assert(all([layer.trainable for layer in self.top_layers.layers]))

        # Visualize which layers are gonna be trained
        self.print_layers_info()

        callbacks_list = []
        trainable_layers = sum([int(layer.trainable) for layer in self.model.layers])
        # Set up save checkpoints for model's weights
        weights_name = self.pretrained_arch + '.' + str(
            trainable_layers) + ".e{epoch:03d}-tloss{loss:.4f}-vloss{val_loss:.4f}.hdf5"
        weights_path = os.path.join(WEIGHTS_OUTPUT_DIR, weights_name)
        callbacks_list.append(keras.callbacks.ModelCheckpoint(
            weights_path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            mode='min'))

        # Set up tensorboard logs
        callbacks_list.append(keras.callbacks.TensorBoard(
            log_dir=TENSORBOARD_LOGS_DIR,
            histogram_freq=1,
            write_graph=True,
            write_images=True))

        # TODO handle usage of class freq weights or balanced batch generators

        class_weights = {
            0: 1 - 0.18,
            1: 1 - 0.55,
            2: 1 - 0.27}

        # class_weights = None

        # Fix needed https://github.com/fchollet/keras/issues/5475
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # Train
        total_unique_images = 1481.
        self.model.fit_generator(
            generator=self.generators['train'],
            steps_per_epoch=int(0.75 * total_unique_images / batch_size),
            epochs=epochs,
            validation_data=self.generators['validate'],
            validation_steps=int(0.25 * total_unique_images / batch_size),
            class_weight=class_weights,
            workers=10,
            callbacks=callbacks_list)
        return

    def predict(self, X):
        return self.model.predict(X, batch_size=1)