import cv2
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
image_path = "/home/pieter/projects/ISMI-Project/data/images/raw/train/Type_1/2920.jpg"

#line 8 and 9 are for testing purposes. 
img = misc.imread(image_path)
shp = img.shape

"""
This methods must go in generator.py.
The call cv2.resize in line 118, def paths_to_images should be
replaced with a call to the definition below. 
First pad the image in the original size (either add black bars to top and
bottom or right and left. 
After padding the image should be a square (same number of pixels for both
dimension). 
After padding the images should be resized (you can call opencv for this). 

"""
def padimage(self, img, dsize=(255,255)):
    shp = img.shape
    if shp[0] > shp[1]:
        to_pad = shp[0] - shp[1]
        Pad = round(float(to_pad/2))
        image = np.zeros(shp[0], shp[0])
	x=0
	y=Pad
	wall[x:x+shp[0], y:y+shp[1]] = shp
    elif shp[1] > shp[0]:
        to_pad = shp[1] - shp[0]
	Pad = round(float(to_pad/2))
	x = Pad
	y = 0
	wall[x:x+shp[0], y:y+shp[1]] = shp
    return cv.resize(img, dsize=(255,255))



