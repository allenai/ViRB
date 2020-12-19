import sys
import os
import glob
import random

path = sys.argv[1]
os.makedirs(path+"test/", exist_ok=True)
cats = glob.glob(path + "train/*")

imgs = [im.split("/")[-1] for im in glob.glob(cats[0]+"/*.png")]
random.shuffle(imgs)
split = imgs[int(0.8*len(imgs)):]

for cat in cats:
    os.makedirs(cat.replace("train", "test"), exist_ok=True)
    for img in split:
        os.rename(cat + "/" + img, cat.replace("train", "test") + "/" + img)
