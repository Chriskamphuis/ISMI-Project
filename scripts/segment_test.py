from segmentator import segment_image
from glob import glob
import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm

def main():
    TEST = "../data/images/raw/test"
    TEST_CROP = "../data/images/pre/test"
    test_files = glob(os.path.join(TEST, "*.jpg"))
    for t in tqdm(test_files):
        file_name = t
        file_save_name = file_name.replace("raw", "pre")
        img = cv2.imread(t)
        img = segment_image(img)
        cv2.imwrite(file_save_name, img)
        

if __name__=="__main__":
    main()
