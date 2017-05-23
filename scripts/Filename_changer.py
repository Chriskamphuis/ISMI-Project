import os
import random

ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TYPE_1 = os.path.join(TRAIN_DIR, "Type_1")
TYPE_2 = os.path.join(TRAIN_DIR, "Type_2")
TYPE_3 = os.path.join(TRAIN_DIR, "Type_3")

os.chdir(TYPE_1)
for num, filename in enumerate(os.listdir(os.getcwd()), start= 1):
    fname, ext = filename, ''
    if '.' in filename:
        fname, ext = filename.split('.')
    os.rename(filename, fname + '_1' + '.' + ext) 

os.chdir(TYPE_2)
for num, filename in enumerate(os.listdir(os.getcwd()), start= 1):
    fname, ext = filename, ''
    if '.' in filename:
        fname, ext = filename.split('.')
    os.rename(filename, fname + '_1' + '.' + ext) 
    
os.chdir(TYPE_3)
for num, filename in enumerate(os.listdir(os.getcwd()), start= 1):
    fname, ext = filename, ''
    if '.' in filename:
        fname, ext = filename.split('.')
    os.rename(filename, fname + '_1' + '.' + ext) 