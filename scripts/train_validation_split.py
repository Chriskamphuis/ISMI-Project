import os
import random


#Directory settings
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TYPE_1 = os.path.join(TRAIN_DIR, "Type_1")
TYPE_2 = os.path.join(TRAIN_DIR, "Type_2")
TYPE_3 = os.path.join(TRAIN_DIR, "Type_3")
#print(TYPE_1)

TYPE_1_1 = os.path.join(DATA_DIR, "Type_1")
TYPE_2_1 = os.path.join(DATA_DIR, "Type_2")
TYPE_3_1 = os.path.join(DATA_DIR, "Type_3")
#print(TYPE_1_1)

#Reading file names from directories
list_type1 = os.listdir(TYPE_1)
list_type2 = os.listdir(TYPE_2)
list_type3 = os.listdir(TYPE_3)
list_type1_1 = os.listdir(TYPE_1_1)
list_type2_1 = os.listdir(TYPE_2_1)
list_type3_1 = os.listdir(TYPE_3_1)

#print(list_type1)

#Shuffling the lists
random.shuffle(list_type1)
random.shuffle(list_type2)
random.shuffle(list_type3)
random.shuffle(list_type1_1)
random.shuffle(list_type2_1)
random.shuffle(list_type3_1)

#Calculating 30%
#print(list_type1)
x1 = len(list_type1)*0.3
#print(x1)
x2 = len(list_type2)*0.3
#print(x2)
x3 = len(list_type3)*0.3
#print(x3)
x1_1 = len(list_type1_1)*0.3
#print(x1_1)
x2_1 = len(list_type2_1)*0.3
#print(x2_1)
x3_1 = len(list_type3_1)*0.3
#print(x3_1)

#Creating validation set
validation_list_type1 = list_type1[:75]
#print(validation_list_type1)
validation_list_type2 = list_type2[:234]
#print(validation_list_type2)
validation_list_type3 = list_type3[:135]
#print(validation_list_type3)
validation_list_type1_1 = list_type1_1[:357]
#print(validation_list_type1_1)
validation_list_type2_1 = list_type2_1[:1070]
#print(validation_list_type2_1)
validation_list_type3_1 = list_type3_1[:593]
#print(validation_list_type3_1)
