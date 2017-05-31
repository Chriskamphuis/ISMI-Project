import sys
import os
from network import Network, Temp
from generator import BatchGenerator

###
#TOP LAYERS TRAINING
###
'''
network = Network(arch = 'xception')
network.freeze_all_pretrained_layers()
train_generator, val_generator = Temp().get_dummy_generators()
network.set_train_val_generators(train_generator, val_generator)
network.compile(finetuning = False)
network.train(epochs = 100, batch_size = 32)
'''
###
#FINE TUNING LAST 60% OF LAYERS
###
'''
network = Network(pretrained_arch = 'vgg16')
network.freeze_all_pretrained_layers()
network.unfreeze_last_pretrained_layers(percentage = 0.5)
train_generator, val_generator = Temp().get_dummy_generators()
network.set_train_val_generators(train_generator, val_generator)
network.compile(finetuning = False)
network.train(epochs = 100, batch_size = 32)
'''
###
#FINE TUNING ALL LAYERS
###
'''
#network = Network(pretrained_arch = 'xception',
#                  input_weights_name = 'xception.82.e002-tloss0.6390-vloss0.9963.hdf5')
#network.freeze_all_pretrained_layers()
#network.unfreeze_last_pretrained_layers(percentage = 0.6)
network = Network(pretrained_arch = 'xception')
train_generator, val_generator = Temp().get_dummy_generators()
network.set_train_val_generators(train_generator, val_generator)
network.compile(finetuning = True)
network.train(epochs = 5000, batch_size = 32)
'''
###
#SIMPLE NETWORK
###
'''
network = Network(arch = 'simple_cnn_1')
train_generator, val_generator = Temp().get_dummy_generators()
network.set_train_val_generators(train_generator, val_generator)
network.compile(finetuning = False)
network.train(epochs = 100, batch_size = 32)
'''



g = BatchGenerator(source = 'pre')
train_filepaths, train_filetargets, val_filepaths, val_filetargets = g.get_splitted_paths_from_csv(use_additional = False)
train_generator = g.generate(data = train_filepaths, labels = train_filetargets, batch_size = 32, shuffle = True)
val_generator = g.generate(data = val_filepaths, labels = val_filetargets, batch_size = 32, shuffle = True)
network = Network(arch = 'vgg16')
network.freeze_all_pretrained_layers()
network.set_train_val_generators(train_generator, val_generator)
network.compile(finetuning = False)
network.train(epochs = 100, batch_size = 32)

