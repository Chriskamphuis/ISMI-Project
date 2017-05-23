import os
import random


#Directory settings
ROOT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")

TYPE_1 = os.path.join(DATA_DIR, "Type_1")
TYPE_2 = os.path.join(DATA_DIR, "Type_2")
TYPE_3 = os.path.join(DATA_DIR, "Type_3")

list_type1 = os.listdir(TYPE_1)
list_type2 = os.listdir(TYPE_2)
list_type3 = os.listdir(TYPE_3)

random.shuffle(list_type1)
random.shuffle(list_type2)
random.shuffle(list_type3)

num_folds = 10
subset_size = len(list_type3)/num_folds
for i in range(num_folds):
    testing_this_round = list_type3[i*subset_size:][:subset_size]
    training_this_round = list_type3[:i*subset_size] + list_type3[(i+1)*subset_size:]
    print(testing_this_round)
