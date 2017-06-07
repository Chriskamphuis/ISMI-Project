import os
import numpy as np
import pandas as pd
import scipy.misc
import cv2
import glob  
from sklearn.preprocessing import LabelBinarizer
from augmenter import chain_augmenters
import random
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input

IMAGES_FOLDER_PATH = os.path.join('..','data','images')
RAW_PATH = os.path.join(IMAGES_FOLDER_PATH,'raw')
PRE_TRAIN_PATH = os.path.join(IMAGES_FOLDER_PATH,'pre-old','train')


class BatchGenerator(object):
        '''
        Generate batches of data. I have decided not to keep all the images in
        memory but the image paths. Now it is reading FULL raw images and thus
        it resizes them in order to stack them and build the batches (so its slow).
        If we are going to segment images offline maybe all of them cam fit in memory.
        '''

        def __init__(self, source):
            if source == 'raw':
                self.images_source = RAW_PATH
            elif source == 'pre':
                self.images_source = PRE_TRAIN_PATH 
            self.augmenter = chain_augmenters(flip=True,
                                              noise=False,
                                              smooth=True,
                                              blur_interval=0.3,
                                              rotate=False,
                                              max_angle=5,
                                              gamma=True,
                                              zoom=True,
                                              transform=False)
            return
        
        def get_splitted_paths_from_csv(self, use_additional):
            '''
            Returns the csv file and returns two lists (train and val) with
            image paths and another two with its respective the classes (0,1,2)
            '''
            #Read CSV
            validation_split_df = pd.read_csv(os.path.join(IMAGES_FOLDER_PATH,'validation_split.csv')).rename(columns = {'Type_1_1':'Type_1_extra','Type_2_1':'Type_2_extra','Type_3_1':'Type_3_extra'})
            #validation_split_df.info()
            validation_split_df = validation_split_df.applymap(lambda x: x.strip() if pd.notnull(x) else x)
            #print validation_split_df.head()
            classes = [0,1,2]
            image_folders = ['Type_1','Type_2','Type_3']

            image_folder_paths = [os.path.join(self.images_source,image_folder) for image_folder in image_folders]
            
            #Generate image paths for VALIDATION from CSV
            print('Validation split:')
            val_filetargets, val_filepaths = [], []
            for c,image_folder in zip(classes, image_folders):
                column = list(validation_split_df[image_folder].dropna())
                val_filepaths += [os.path.join(self.images_source,image_folder,filename) for filename in column]
                val_filetargets += list(np.repeat(c,len(column)))
                #print len(column),c
                if use_additional:
                    column = list(validation_split_df[image_folder+'_extra'].dropna())
                    val_filepaths += [os.path.join(self.images_source,image_folder+'_extra',filename) for filename in column]
                    val_filetargets += list(np.repeat(c,len(column)))
                print('\t',len(column),'of type',c+1)

            #Check that references in CSV exist in disk
            assert all([os.path.exists(path) for path in val_filepaths])

            #Generate image paths for TRAINING from {ALL - VALIDATION}
            print('Training split:')
            train_filetargets, train_filepaths = [], []
            for c,image_folder in zip(classes, image_folders):
                all_folder_images = glob.glob(os.path.join(self.images_source,image_folder,'*'))
                if use_additional:
                    all_folder_images += glob.glob(os.path.join(self.images_source,image_folder+'_extra','*'))
                #Train = All - Validation
                train_folder_images = [path for path in all_folder_images if path not in val_filepaths]
                train_filepaths += train_folder_images
                train_filetargets += list(np.repeat(c,len(train_folder_images)))
                print('\t',len(train_folder_images),'of type',c+1)

            #Check that unferred references in CSV exist in disk
            assert all([os.path.exists(path) for path in train_filepaths])

            '''self.train_filepaths = train_filepaths
            self.train_filetargets = train_filetargets
            self.val_filepaths = val_filepaths
            self.val_filetargets = val_filetargets'''
            return train_filepaths, train_filetargets, val_filepaths, val_filetargets
   
        def generate(self, data, labels, batch_size=32,
                     balanced=True):
            '''
            Generate batches of data images taking as input
            a list of image paths and labels
            '''
            encoder = LabelBinarizer().fit(np.array([0,1,2]))
            while True:
                data, labels = self.shuffle(data, labels)
                batches = int(len(data)/batch_size)
                #print batches                
                for batch in range(batches):
                    #print batch
                    self.augmenter.randomize()
                    x_image_paths = data[batch*batch_size:(batch+1)*batch_size]
                    x = self.paths_to_images(x_image_paths)
                    y = np.array(labels[batch*batch_size:(batch+1)*batch_size])
                    if len(y) != batch_size:
                        break
                    y = encoder.transform(y)
                    #x = preprocess_input(x)
                    yield((x, y))
                    
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
            
        def paths_to_images(self,paths):
            '''
            Converts a list of imagepaths to a list of images
            '''
            #images = [scipy.misc.imread(path) for path in paths]
            images = [Image.open(path) for path in paths]
            #TODO remove resize when reading presegmented images
            images = [self.random_padding(img) for img in images]
            images = [cv2.resize(img, dsize=(224,224)) for img in images]
            #images_ = []
            #for img in images:
                #self.augmenter.randomize()
                #images_.append(self.augmenter.augment(img)[0])
            #images = images_
            images = [self.augmenter.augment(img)[0] for img in images]
            
            images = [img[np.newaxis,:,:,:] for img in images]
            #images = [img[0][np.newaxis,:,:,:] for img in images]
            #images = [img[:255,:][np.newaxis,:,:,:] for img in images]
            #images = [img[np.newaxis,:,:,:] for img in images]
            result = np.concatenate(images) 
            return result
        
        def shuffle(self, data, labels):
            '''
            Shuffles data keeping <image,label> pairs
            together
            '''
            indices = np.arange(len(data))
            np.random.shuffle(indices)
            return list(pd.Series(data)[indices]), list(pd.Series(labels)[indices])

        def get_splitted_paths_from_folders(self, val_perc, use_additional):
            classes = [0,1,2]
            train_filepaths, train_filetargets, val_filepaths, val_filetargets = [], [], [], []
            #folders = ['Type_'+str(clss) for clss in classes]
            #if use_additional:
            #for clss in classes:
            #        folders.append('Type_'+str(clss)+'_extra')
            for clss in classes:
                folders = ['Type_'+str(clss+1)]
                if use_additional:
                    folders.append(folders[0]+'_extra')
                for folder_name in folders: 
                    filepaths = glob.glob(os.path.join(PRE_TRAIN_PATH,folder_name,'*.jpg'))
                    np.random.shuffle(filepaths)
                    n = int(val_perc * len(filepaths))
                    clss_train_filepaths = filepaths[:len(filepaths)-n]
                    clss_val_filepaths = filepaths[len(filepaths)-n:]
                    clss_train_filetargets = np.repeat(clss,len(clss_train_filepaths))
                    clss_val_filetargets = np.repeat(clss,len(clss_val_filepaths))
                    print(folder_name,":\n\t",len(clss_train_filepaths)," for training\n\t",len(clss_val_filepaths),"for validation\n\ttotal",len(filepaths))
                    #print(clss_train_filetargets)
                    #print(clss_val_filetargets)
                    train_filepaths += clss_train_filepaths
                    train_filetargets.append(clss_train_filetargets)
                    val_filepaths += clss_val_filepaths
                    val_filetargets.append(clss_val_filetargets)
            val_filetargets = np.hstack(val_filetargets)
            train_filetargets = np.hstack(train_filetargets)
            
            train_filepaths, train_filetargets = self.shuffle(train_filepaths, train_filetargets)
            val_filepaths, val_filetargets = self.shuffle(val_filepaths, val_filetargets)
            print(len(val_filetargets))
            print(len(train_filetargets))
            print(len(train_filepaths),len(val_filepaths))
            #print("Generating splits from folders:",folders)
            return train_filepaths, train_filetargets, val_filepaths, val_filetargets
            
            


