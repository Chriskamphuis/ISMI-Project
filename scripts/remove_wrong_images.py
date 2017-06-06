import os
import pandas as pd

df = pd.read_csv(os.path.join("..","data","images","wrong_images.csv"))


for i, folder in enumerate(df["Folder"]):
    name = str(df["Name"].get_value(i))+".jpg"
    os.remove(os.path.join("..","data","images","raw","train",folder,name))