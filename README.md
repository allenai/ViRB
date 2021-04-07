# pyVTAB

pyVTAB is a framework for evaluating the quality of representations 
learned by visual encoders on a variety of downstream tasks. It is the 
codebase used by the paper [Contrasting Contrastive Self-Supervised 
Representation Learning Models](https://arxiv.org/pdf/2103.14005.pdf).
As this is a tool for evaluating the learned representations, it is 
designed to freeze the encoder weights and only train a small end task
network using latent representations on the train set for each
task and evaluate it on the test set for that task. To speed this process 
up, the train and test set are pre encoded for most of the end tasks and 
stored in GPU memory for efficient usage. Fine tuning the encoder is also 
supported but takes significantly more time.
pyVTAB is fully implemented in pyTorch and automatically scales to as many 
GPUs as are available on your machine. It has support for evaluating 
any pyTorch model architecture on a select subset of tasks.

## Installation

To install the codebase simply clone this repository from github and run setup:
```shell script
git clone https://github.com/klemenkotar/pyVTAB
cd pyVTAB
pip install -r requirements.txt
```

## Quick Start

For a quick starting example we will train an end task network on the simple
CalTech classification task using the SWAV 800 encoder.

First we need to download the encoder:
```shell script
mkdir pretrained_weights
wget https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAV_800.pt pretrained_weights/
```

Then we need to download the CalTech dataset from 
[here](http://www.vision.caltech.edu/Image_Datasets/Caltech101/) and extract it
into a directory called `data/caltech/`.


## Dataset Download

To run the full suit of end tasks we need to download all the associated 
datasets. In general all the datasets should be stored in a folder called `data/`
inside the root project directory. Bellow is a table of links where the data can 
be downloaded and the names of directories they should be downloaded to.

| Dataset Name  | Dataset Size | Download Link|
|---------------|--------------|--------------|
 ImageNet Cls. | 1,281,167 | [Link](http://www.image-net.org/download)
 Pets Cls. | 3,680  | [Link](http://www.image-net.org/download)
 CalTech Cls. |  3,060  | [Link](http://www.image-net.org/download) 
 CIFAR-100 Cls. |  50,000 | [Link](http://www.image-net.org/download) 
 SUN Scene Cls. |  87,003 | [Link](http://www.image-net.org/download) 
 Eurosat Cls. |  21,600 | [Link](http://www.image-net.org/download) 
 dtd Cls. |  3,760 | [Link](http://www.image-net.org/download)
 Kinetics Action Pred. |  50,000 | [Link](http://www.image-net.org/download) 
 CLEVR Count | 70,000 | [Link](http://www.image-net.org/download) 
 THOR Num. Steps | 60,000 | [Link](http://www.image-net.org/download) 
 THOR Egomotion | 60,000 | [Link](http://www.image-net.org/download) 
 nuScenes Egomotion | 28,000 | [Link](http://www.image-net.org/download) 
 Cityscapes Seg. | 3,475 | [Link](http://www.image-net.org/download)
 Pets Instance Seg. | 3,680 | [Link](http://www.image-net.org/download) 
 EgoHands Seg. | 4,800 |  [Link](http://www.image-net.org/download) 
 THOR Depth | 60,000 | [Link](http://www.image-net.org/download) 
 Taskonomy Depth | 39,995 | [Link](http://www.image-net.org/download) 
 NYU Depth | 1,159 | [Link](http://www.image-net.org/download) 
 NYU Walkable | 1,159  | [Link](http://www.image-net.org/download) 
 KITTI Opt. Flow | 200 | [Link](http://www.image-net.org/download) 

## Pre-trained Models

As part of our paper we trained several new encoders using a combination of training 
algorithms and datasets. Bellow is a table containing the download links to the weights.
The weights are stored in standard pyTorch format. To work with this codebase, 
the models should be downloaded into a directory called `pretrained_weights/` inside the
root project directory.


| Encoder Name | Method | Dataset | Dataset Size | Number of Epochs | Trained by us|Link| 
|--------------|--------|---------|--------------|------------------|--------------|----|
|SwAV ImageNet 800 | SwAV | ImageNet | 1.3M | 800 | No| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAV_800.pt)|
|SwAV ImageNet 200 | SwAV | ImageNet | 1.3M | 200 | No| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAV_200.pt)|
|SwAV ImageNet 100 | SwAV | ImageNet | 1.3M | 100 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAV_100.pt)|
|SwAV ImageNet 50 | SwAV | ImageNet | 1.3M | 50 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAV_50.pt)|
|SwAV Half ImageNet 200 | SwAV | ImageNet-1/2 | 0.5M | 200 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAVHalfImagenet.pt)|
|SwAV Half ImageNet 100 | SwAV | ImageNet-1/2 | 0.5M | 100 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAVHalfImagenet_100.pt)|
|SwAV Quarter ImageNet 200 | SwAV | ImageNet-1/4 | 0.25M | 200 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAVQuarterImagenet.pt)|
|SwAV Linear Unbalanced ImageNet 200 | SwAV | ImageNet-1/2-Lin | 0.5M | 200 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAVUnbalancedImagenet.pt)|
|SwAV Linear Unbalanced ImageNet 100 | SwAV | ImageNet-1/2-Lin | 0.5M | 100 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAVUnbalancedImagenet_100.pt)|
|SwAV Log Unbalanced ImageNet 200 | SwAV | ImageNet-1/4-Log | 0.25M | 200 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAVLogImagenet.pt)|
|SwAV Places 200 | SwAV | Places | 1.3M | 200 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAVPlaces.pt)|
|SwAV Kinetics 200 | SwAV | Kinetics | 1.3M | 200 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAVKinetics.pt)|
|SwAV Taskonomy 200 | SwAV | Taskonomy | 1.3M | 200 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAVTaskonomy.pt)|
|SwAV Combination 200 | SwAV | Combination | 1.3M | 200 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/SWAVCombination.pt)|
|MoCov2 ImageNet 800 | MoCov2 | ImageNet | 1.3M | 800 | No|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2_800.pt)|
|MoCov2 ImageNet 200 | MoCov2 | ImageNet | 1.3M | 200 | No|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2_200.pt)|
|MoCov2 ImageNet 100 | MoCov2 | ImageNet | 1.3M | 100 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2_100.pt)|
|MoCov2 ImageNet 50 | MoCov2 | ImageNet | 1.3M | 50 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2_50.pt)|
|MoCov2 Half ImageNet 200 | MoCov2 | ImageNet-1/2 | 0.5M | 200 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2HalfImagenet.pt)|
|MoCov2 Half ImageNet 100 | MoCov2 | ImageNet-1/2 | 0.5M | 100 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2HalfImagenet_100.pt)|
|MoCov2 Quarter ImageNet 200 | MoCov2 | ImageNet-1/4 | 0.25M | 200 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2QuarterImagenet.pt)|
|MoCov2 Linear Unbalanced ImageNet 200 | MoCov2 | ImageNet-1/2-Lin | 0.5M | 200 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2UnbalancedImagenet.pt)|
|MoCov2 Linear Unbalanced ImageNet 100 | MoCov2 | ImageNet-1/2-Lin | 0.5M | 100 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2UnbalancedImagenet_100.pt)|
|MoCov2 Log Unbalanced ImageNet 200 | MoCov2 | ImageNet-1/4-Log | 0.25M | 200 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2LogImagenet.pt)|
|MoCov2 Places 200 | MoCov2 | Places | 1.3M | 200 | Yes| [Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2Places.pt)|
|MoCov2 Kinetics 200 | MoCov2 | Kinetics | 1.3M | 200 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2Kinetics.pt)|
|MoCov2 Taskonomy 200 | MoCov2 | Taskonomy | 1.3M | 200 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2Taskonomy.pt)|
|MoCov2 Combination 200 | MoCov2 | Combination | 1.3M | 200 | Yes|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov2Combination.pt)|
|MoCov1 ImageNet 200 | MoCov1 | ImageNet | 1.3M | 200 | No|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/MoCov1_200.pt)|
|PIRL ImageNet 800 | PIRL | ImageNet | 1.3M | 800 | No|[Link](https://prior-model-weights.s3.us-east-2.amazonaws.com/contrastive_encoders/PIRL_800.pt)|

## End Task Training

pyVTAB supports 20 end task that are classified as `Image-level` or `Pixelwise`
depending on the output modality of the task. Furthermore each task is also 
classified as either semantic or structural. Bellow is an illustration of the space of
our tasks. For further details please see [Contrasting Contrastive Self-Supervised 
Representation Learning Models](https://arxiv.org/pdf/2103.14005.pdf).

![Tasks](assets/images/tasks.png)

After installing the codebase and downloading the datasets and pretrained models we are 
ready to run our experiments. To reproduce every experiment in the paper run:
```shell script
python main.py --experiment_list=configs/experiment_lists/all.yaml 
--vtab_configs=configs/vtab_configs/all.yaml
```
`WARNING:` this will take well over 1000 GPU hours to train so we suggest training a 
subset instead. We can see the results of all these training runs summarized in the 
graph bellow.

![Results](assets/images/all_results.png)
*Correlation of end task performances with ImageNet classification accuracy.
The plots show the end task performance against the ImageNet top-1 accuracy 
for all end tasks and encoders. Each point represents a different encoder 
trained with different algorithms and datasets. This reveals the lack of a 
strong correlation between the performance on ImageNet classification and 
tasks from other categories.*

To specify which task we want to train we create a vtab_config yaml file which defines
the task name and training configuration. The file `configs/vtab_configs/all.yaml` 
contains configurations for every task supported by this package so it is a good 
starting point. We can select only a few tasks to train and comment out the other 
configurations.

To specify which weights we want to use we specify an experiment list file. The 
file `configs/experiment_lists/all.yaml` contains all the model weights provided 
by this repository. We can select only a few models to train and comment out the other 
configurations. Alternatively we can add in new weights and add them to the list. 
All we have to do is make sure the weights are for a ResNet50 model stored in the 
standard pyTorch weight file.


### Hyperparameter Search
One feature offered by this codebase is the ability to train the end task networks using
several sets of optimizers, schedulers and hyperparameters. For the Image-level tasks 
(which are encodable), the dataset will get encoded only once and then a model using 
each set of hyperparameters will get trained (to improve efficiency). 

An example of a grid search configuration can be found in 
`configs/vtab_configs/imagenet_grid_search.yaml`, and it looks like this:
```yaml
Imagenet:
 task: "Imagenet"
 training_configs:
   adam-0.0001:
     optimizer: "adam"
     lr: 0.0001
   adam-0.001:
     optimizer: "adam"
     lr: 0.001
   sgd-0.01-StepLR:
     optimizer: "sgd"
     lr: 0.01
     scheduler:
       type: "StepLR"
       step_size: 50
       gamma: 0.1
   sgd-0.01-OneCycle:
     optimizer: "sgd"
     lr: 0.01
     scheduler:
       type: "OneCycle"
   sgd-0.01-Poly:
     optimizer: "sgd"
     lr: 0.001
     scheduler:
       type: "Poly"
       exponent: 0.9
 num_epochs: 100
 batch_size: 32
```
We spoecify each training config as a YAML object. The `"sgd"` and `"adam"` optimizers 
are supported as well as the `"StepLR"`, `"OneCycle"` and `"Poly"` schedulers from 
pyTorch's `optim` package. All schedulers are compatible with all of the optimizers.

To execute this ImageNet grid search run:
```shell script
python main.py --experiment_list=configs/experiment_lists/swav.yaml 
--vtab_configs=configs/vtab_configs/imagenet_grid_search.yaml
```

### Testing Only Datasets

One aditional feature this codebase supports is datasets that are "eval only" and use
a task head trained on a different task. The only currently supported example is 
ImageNet v2. To test the SWAV 800 model on ImageNetv2 first train at least one 
ImageNet end task head on SWAV 800 then run the following command:
```shell script
python main.py --experiment_list=configs/experiment_lists/swav.yaml 
--vtab_configs=configs/vtab_configs/imagenetv2.yaml
```

## Custom Models

All the encoders in the tutorials thus far have used the ResNet50 architecture, but we also 
support using custom encoders. 

All of the Image-level tasks require the encoder outputs a 
dictionary with the key "embedding" mapping to a pyTorch tensor of size `NxD` where `N` is the
batch size and `D` is the arbitrary embedding size. 

All of the Pixelwise tasks require that the encoders output a dictionary with a tensor
for the representation after every block. In practice this means that the model needs to output
5 tensors of sizes corresponding to the outputs of a ResNet50 `conv`, `block1`, `block2`, `block3`
and `block4` layers.

To use a custom model simply modify `main.py` by replacing `ResNet50Encoder` with any encoder with
the outputs mentioned above.

## Citation
```
@article{kotar2021contrasting,
  title={Contrasting Contrastive Self-Supervised Representation Learning Models},
  author={Klemen Kotar and Gabriel Ilharco and Ludwig Schmidt and Kiana Ehsani and Roozbeh Mottaghi},
  year={2021},
}
```
