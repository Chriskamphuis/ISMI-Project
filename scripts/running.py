import sys
import os
from network import Network, Temp
from generator import BatchGenerator
import scipy.misc

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



arch = 'inception'
batch_size = 32
#input_weights_name = None
#input_weights_name = 'vgg19.3.e050-tloss2.8528-vloss0.9718.hdf5'
#input_weights_name = 'vgg16.3.e034-tloss2.7046-vloss1.0362.hdf5'
#input_weights_name = 'xception.3.e016-tloss2.2526-vloss0.9234.hdf5'
#input_weights_name = 'resnet.3.e005-tloss2.7884-vloss1.3572.hdf5'
input_weights_name = 'inception.3.e005-tloss3.1504-vloss1.0324.hdf5'
g = BatchGenerator(source = 'pre')
train_filepaths, train_filetargets, val_filepaths, val_filetargets = g.get_splitted_paths_from_folders(val_perc = 0.2, use_additional = False)
train_generator = g.generate(data = train_filepaths, labels = train_filetargets, batch_size = batch_size)
val_generator = g.generate(data = val_filepaths, labels = val_filetargets, batch_size = batch_size)
'''
network = Network(arch = arch,
                  input_weights_name = input_weights_name)
network.freeze_all_pretrained_layers()
network.set_train_val_generators(train_generator, val_generator)
network.unfreeze_last_pretrained_layers(percentage = 0.7)
network.compile(finetuning = input_weights_name == None)
network.train(epochs = 500, batch_size = batch_size)
'''


for a in range(10):
    x_batch, y_batch = train_generator.__next__()
    print(x_batch.shape)
    print(y_batch.shape)
    for i in range(20):
        img = x_batch[i]
        #print(img.mean(),img.max(),img.min())
        #print(img)
        scipy.misc.imsave('delete/'+str(i)+str(a)+str(y_batch[i])+'.jpg',img)


for a in range(10):
    x_batch, y_batch = val_generator.__next__()
    print(x_batch.shape)
    print(y_batch.shape)
    for i in range(20):
        img = x_batch[i]
        #print(img.mean(),img.max(),img.min())
        #print(img)
        scipy.misc.imsave('delete/'+str(i)+str(a)+str(y_batch[i])+'.jpg',img)
