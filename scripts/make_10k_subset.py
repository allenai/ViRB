import glob
from PIL import Image
import random
import os


MAX_IMGS = 10000
ROOT = '../data/omni_10k/'
PATHS = {
    'Caltech': '../data/caltech-101/train/*/*.jpg',
    'Cityscapes': '../data/cityscapes/leftImg8bit/train/*/*.png',
    'CLEVR': '../data/CLEVR/images/train/*.png',
    'dtd': '../data/dtd/train/*/*.jpg',
    'Egohands': '../data/egohands/images/*/*.jpg',
    'Eurosat': '../data/eurosat/train/*/*.jpg',
    'ImageNet': '../data/imagenet/train/*/*.JPEG',
    'Kinetics': '../data/kinetics400/train/*/*.jpg',
    'KITTI': '../data/KITTI/training/image_2/*.png',
    'nuScenes': '../data/nuScenes/samples/CAM_FRONT/*.jpg',
    'NYU': '../data/nyu/train/images/*.png',
    'Pets': '../data/pets/train/*/*.jpg',
    'SUN397': '../data/SUN397/train/*/*.jpg',
    'Taskonomy': '../data/taskonomy/train/rgb/*/*.png',
    'Thor': '../data/thor_action_prediction/train/*/*.jpg'
}

data = []
for key in PATHS:
    os.makedirs(ROOT+key,exist_ok=True)
    path = PATHS[key]
    imgs = glob.glob(path)
    imgs.sort()
    random.seed(1999)
    random.shuffle(imgs)
    print("Making %s Dataset of Size %d" % (key, min(len(imgs), MAX_IMGS)))
    for i in range(min(len(imgs), MAX_IMGS)):
        # data.append(imgs[i])
        im = Image.open(imgs[i]).resize((224, 224))
        im.save("%s/%s/%d.png" % (ROOT, key, i))
