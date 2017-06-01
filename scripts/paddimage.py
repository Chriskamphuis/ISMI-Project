import cv2
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
image_path = "/home/zina/Downloads/test.jpg"

#line 8 and 9 are for testing purposes. 


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
        Pad = int(round(float(to_pad/2)))
        image = np.zeros((shp[0], shp[0], shp[2]))
	print image.shape
	print img.shape
	x=0
	y=Pad
	image[x:x+shp[0], y:y+shp[1]] = img
    elif shp[1] > shp[0]:
        to_pad = shp[1] - shp[0]
	Pad = int(round(float(to_pad/2)))
        image = np.zeros((shp[1], shp[1], shp[2]))
	print image.shape
	print img.shape
	x = Pad
	y = 0
	image[x:x+shp[0], y:y+shp[1], :] = img
    return cv2.resize(image, dsize=(255,255))

img = cv2.imread(image_path)
newimg = padimage(img)
cv2.imwrite('padded_test.png',newimg)

