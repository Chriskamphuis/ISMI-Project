import argparse
import os
import numpy as np
import datetime
import pandas as pd
from pytz import timezone
from scipy.misc import imread, imresize
from network import Network

SUBMISSION_DIR = os.path.join('..','data','submissions')
TEST_DIR = os.path.join('..','data','images','raw','test') #!!SHOULD BE REPRLACED WITH THE PREPOCESSED FOLDER IF THE IMAGES EXIST!!
CLASSES = ['Type_1','Type_2','Type_3']

'''
Script needs the location of the weights and the model used for the weights, test directory is optional
'''
parser = argparse.ArgumentParser()
parser.add_argument('--weights', help="Name of the wanted weights saved in the weights folder", required=True)
parser.add_argument('--model', help="Name of the architecture to be used", required=True)
parser.add_argument('--test', help="Directory of the test data", default=TEST_DIR)

args = parser.parse_args()

output_weights_name = str(args.weights)
model_name = str(args.model)
TEST_DIR = str(args.test)


def load_test_data(loadPath):
    '''
    Load test data
    '''
    images = []
    image_names = []
    photos = os.listdir(loadPath)
    for photo in photos:
        if (not 'Thumbs.db' in photo):
            image = imread(os.path.join(loadPath,photo))
            image = imresize(image,size=(224,224,3))
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
    image_names,images = load_test_data(TEST_DIR)
    network = Network(model_name,output_weights_name)
    network.compile(False)
    predictions = network.predict(images)
    image_names, images = sort(image_names, predictions)
    write_submission_file(image_names,predictions)
