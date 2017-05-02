import argparse
import os
import numpy as np
import datetime
from pytz import timezone
from convnetskeras.convnets import preprocess_image_batch
from SimpleNetwork import Network

SUBMISSION_DIR = os.path.join('..','data','submissions')
TEST_DIR = os.path.join('..','data','images','raw','test')
CLASSES = ['Type_1','Type_2','Type_3']

'''
Script needs the location of the weights and the model used for the weights, test directory and batch size are optional
'''
parser = argparse.ArgumentParser()
parser.add_argument('--weights', help="Name of the wanted weights saved in the output folder", required=True)
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
            prep = preprocess_image_batch([loadPath + '/' + photo], img_size=(224, 224), color_mode="bgr")
            prep = prep.transpose([0,2,3,1])
            print prep.shape
            images.extend(prep)
            image_names.append(photo)
            break
    return (image_names, np.array(images))


def get_timestamp():
    '''
    Create a timestamp to avoid overwriting of submission files and to distinguish the different submission files
    '''
    amsterdam_tz = timezone('Europe/Amsterdam')
    time = amsterdam_tz.localize(datetime.datetime.now(), is_dst=False)
    return time.strftime('%Y-%M-%d-%H-%M-%S')


def list_to_string(l):
    '''
    Convenience function: convert list to a comma separated string
    '''
    string = ""
    for e in l:
        string += ',' + str(e).strip()
    return string


def write_submission_file(image_names, predictions):
    '''
    Write submission file in the format asked for in the Intel & MobileODT Cervical Cancer Screening challenge on Kaggle
    '''
    output = open(os.path.join(SUBMISSION_DIR,'submission-'+get_timestamp()+'.csv'), 'wb')
    output.write('image_name,' + str(CLASSES).strip('[]').replace("'", "").strip() + '\r\n')
    for i in range(len(image_names)):
        output.write(image_names[i] + list_to_string(predictions[i]) + '\r\n')
    output.close()


if __name__ == "__main__":
    image_names,images = load_test_data(TEST_DIR)
    network = Network(model_name,output_weights_name=output_weights_name)
    network.compile(False)
    predictions = network.predict(images)
    write_submission_file(image_names,predictions)
