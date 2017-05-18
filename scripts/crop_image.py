import pandas as pd
import os
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

RAWDATADIR = os.path.join("..","data","images","raw","train")
PREDATADIR = os.path.join("..","data","images","pre","train")
cropsizes = pd.read_csv(os.path.join("..","notebooks","rectangles.csv"),index_col=False)

data = []
for dirpath, dirnames, filenames in os.walk(RAWDATADIR):
    for filename in [f for f in filenames if f.endswith(".jpg")]:
        data.append(os.path.join(dirpath, filename))

for i,im in enumerate(data):
    im = im.split("/")
    print im
    image = Image.open(os.path.join(RAWDATADIR,im[-2], im[-1]))
    image = ImageOps.fit(image, (192,256)) 
    image = image.crop((cropsizes["x"][i], cropsizes["y"][i], cropsizes["x"][i]+cropsizes["w"][i], cropsizes["y"][i]+cropsizes["h"][i]))
    if not os.path.exists(os.path.join(PREDATADIR,im[-2])):
        os.makedirs(os.path.join(PREDATADIR,im[-2]))
    image.save(os.path.join(PREDATADIR,im[-2],im[-1]))
