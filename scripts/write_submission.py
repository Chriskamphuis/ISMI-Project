import argparse
import os
import numpy as np
import datetime
import pandas as pd
from pytz import timezone
import scipy
from scipy.misc import imread, imresize, imsave
from network import Network
from augmenter import chain_augmenters


SUBMISSION_DIR = os.path.join('..','data','submissions')
TEST_DIR = os.path.join('..','data','images','pre-old','test') #!!SHOULD BE REPRLACED WITH THE PREPOCESSED FOLDER IF THE IMAGES EXIST!!
CLASSES = ['Type_1','Type_2','Type_3']

'''
Script needs the location of the weights and the model used for the weights, test directory is optional
'''
parser = argparse.ArgumentParser()
parser.add_argument('--weights', help="Name of the wanted weights saved in the weights folder", required=True)
parser.add_argument('--model', help="Name of the architecture to be used", required=True)
parser.add_argument('--test', help="Directory of the test data", default=TEST_DIR)

args = parser.parse_args()

input_weights_name = str(args.weights)
arch = str(args.model)
TEST_DIR = str(args.test)
import random

def load_test_data(loadPath):
    '''
    Load test data. We apply the same augmentations as in the training
    process so that we feed the net with what it expects
    '''
    images = []
    image_names = []
    photos = os.listdir(loadPath)
    size = (224,224,3)
    augmenter = chain_augmenters(flip=True,
                                  noise=False,
                                  smooth=True,
                                  blur_interval=0.3,
                                  rotate=False,
                                  max_angle=5,
                                  gamma=True,
                                  zoom=True,
                                  transform=False)
    augmenter.randomize()
    for photo in photos:
        if (not 'Thumbs.db' in photo):
            image = imread(os.path.join(loadPath,photo))
            image = imresize(image,size=size)
            image= augmenter.augment(image)[0]
            #scipy.misc.imsave('/lustre2/0/ismikag2017/ISMI-project/scripts/delete/'+str(random.randint(0,99999))+'.jpg',image)
            images.append(image)
            image_names.append(photo)
    return (image_names, np.array(images))


def get_timestamp():
    '''
    Create a timestamp to avoid overwriting of submission files and to distinguish the different submission files
    '''
    amsterdam_tz = timezone('Europe/Amsterdam')
    time = amsterdam_tz.localize(datetime.datetime.now(), is_dst=False)
    return time.strftime('%Y-%M-%d-%H-%M-%S')

def sort(image_names,predictions):
    '''
    Sorts the files in the right order for the submission file
    '''
    image_numbers = np.asarray([ int(image_name[:-4]) for image_name in image_names])
    total = np.column_stack((image_numbers,predictions))
    total = total[total[:,0].argsort(),:]
    image_names = [str(int(number)) + ".jpg" for number in total[:,0]]
    return image_names, total[:,[1,2,3]]

def write_submission_file(image_names, predictions):
    '''
        Write submission file in the format asked for in the Intel & MobileODT Cervical Cancer Screening challenge on Kaggle
    '''
    submission = pd.DataFrame({"image_name": image_names,
        CLASSES[0]: predictions[:,0],
        CLASSES[1]: predictions[:,1],
        CLASSES[2]: predictions[:,2],
        })
    submission = submission[['image_name', CLASSES[0], CLASSES[1], CLASSES[2]]]
    submission.to_csv(os.path.join(SUBMISSION_DIR,"submission-"+get_timestamp()+".csv"),index=False)


if __name__ == "__main__":
    print('loading test images from ',TEST_DIR)
    image_names,images = load_test_data(TEST_DIR)
    print('loading network ',input_weights_name)
    network = Network(arch,input_weights_name)
    network.compile(False)
    predictions = network.predict(images)
    print('writting submission file')
    image_names, images = sort(image_names, predictions)
    write_submission_file(image_names,predictions)
