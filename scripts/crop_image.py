import numpy as np
import pandas as pd
import os
from PIL import Image

RAWDATADIR = os.path.join("..","data","images","raw")
PREDATADIR = os.path.join("..","data","images","pre")
cropsizes = pd.read_csv(os.path.join("..","notebooks","rectangles.csv"),index_col=False)
data = os.listdir(RAWDATADIR)

print cropsizes["w"]
print cropsizes["h"]
print cropsizes["x"]
print cropsizes["y"]

print np.asarray(cropsizes["x"])

for i,im in enumerate(data):
    image = Image.open(os.path.join(RAWDATADIR,im))
    image = image.crop(cropsizes["x"][i], cropsizes["y"][i], cropsizes["w"][i], cropsizes["h"][i])
    image.save(os.path.join(PREDATADIR,image))