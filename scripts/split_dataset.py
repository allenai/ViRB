import sys
import os
import glob
import random

path = sys.argv[1]
os.makedirs(path+"test/", exist_ok=True)
cats = glob.glob(path + "train/*")
for cat in cats:
    os.makedirs(cat.replace("train", "test"), exist_ok=True)
    imgs = glob.glob(cat+"/*.png")
    random.shuffle(imgs)
    split = imgs[int(0.8*len(imgs)):]
    for img in split:
        os.rename(img, img.replace("train", "test"))
