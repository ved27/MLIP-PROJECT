# Optimizing SSD Multi-Object Detection Model (PyTorch)

## Table of contents
* [Installation](#Installation)
* [Datasets and weights](#Datasets)
* [Demo](#Demo)
* [Training](#Training)
* [Evaluation](#Evaluation)
* [Performance](#Performance)
* [Directory structure](#Directory-structure)
* [References](#References)

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Installation

To install Python dependencies and modules, use <br>
```pip install -r requirements.txt``` <br>

- Choose your environment on the website and install [PyTorch](http://pytorch.org/) by running the appropriate command.
- Clone this repository.
- Install one of Python 3+.
- Then download the dataset by following the [instructions](#datasets) below.

## Datasets and weights

In this project we are using the `2007 and 2012 VOC trainval` dataset in order to train the model. Testing the model is implemented using `2007 VOC test` dataset. You can download the dataset by clicking on the following links. 

[2007 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)


[2012 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) 

Please access the following links and extract the contents to `trained_weights` and `weights` respectively

trained_weights - has trained weights corresponding to experiments performed. 

weights - has pre-trained model weights

[trained_weights](https://drive.google.com/file/d/1TkPI5jLpv2YvMPiDKdqp3TN2E3KJ6IzJ/view?usp=sharing)

[weights](https://drive.google.com/file/d/16f6osiiFouUUUJw0qH_5UtS67VqvuVIs/view?usp=sharing)

 


### Demo

Update ` voc_root ` and ` dataset_root ` in config_vgg16_ssd.py with the location of your downloaded dataset.
`voc_root` -> should contain VOC2007 and VOC2012 directories -> ` PATH_TO/ ` 
and  `dataset_root ` -> `PATH_TO/VOC2007/` 

Make sure you have folders named `trained_weights` and `weights` now

If you want to run a notebook , assign VOC_ROOT to your `PATH_TO/ ` in the notebook.

Run **SSD_Demo-baseline_vgg16-ssd.ipynb** to run baseline vgg16 ssd on an image in `test_images` or a random image of VOC2007 set

Run **SSD_Demo-feature_fused_concat_deconvolution.ipynb** to run vgg16 ssd with feature fusion concat module added

Run **SSD_Demo-feature_fused_eltsum_deconvolution.ipynb** to run vgg16 ssd with feature fusion eltsum module added

Run **SSD_Demo_vgg16-ssd-pyramidal.ipynb** to run vgg16 with modified ssd using pyramidal feature extractor module 

### Training
Models are trained on VOC2012+2007 trainval set 

**PROJ_vgg16_ssd_train-voc2012-2007_feature_fused_concat_deconvolution.ipynb** to resume training of vgg16 ssd with feature fusion concat module 

**PROJ_vgg16_ssd_train-voc2012-2007_feature_fused_eltsum_deconvolution.ipynb** to resume training of vgg16 ssd with feature fusion 
eltsum module

**PROJ_vgg16_ssd_train-voc2012-2007_pyramidal.ipynb** to resume training of vgg16 ssd usign pyramidal feature extractor module

### Evaluation
Evaluate the SSD models on the VOC2007 test set. 
Download the PascalVOC2007 test set using `wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar` and run `tar -xvf VOCtest_06-Nov-2007.tar` in the root directory of the repository. 
Make sure the contents are extracted to `PATH_TO/VOC2007/ ` 

**PROJ_vgg16_ssd_eval_VOC2007testset .ipynb**

**PROJ_vgg16_ssd_eval_VOC2007testset-feature_fused_concat_deconvolution.ipynb** to evaluate  vgg16 ssd with feature fusion concat module 

**PROJ_vgg16_ssd_eval_VOC2007testset-feature_fused_eltsum_deconvolution.ipynb** to evaluate  vgg16 ssd with feature fusion eltsum module 

**PROJ_vgg16_ssd_eval_VOC2007testset-pyramidal.ipynb**  to resume training of vgg16 ssd using pyramidal feature extractor module

### Performance 

| Category    | Baseline (mAP) | SSD with FF concat (mAP) | SSD with FF eltsum module (mAP) | pyramidal feature extractor

| ---------- |  ------------- | ----------------------- - | ------------------------------- | ------------------------

| Evaluation  |   77.47%     |   77.985%                |     77.987%                      |      78.04%


### Directory structure
- architectures
   - FSSD_vgg.py - contains the SSD model with pyramidal feature extractor 
   - base_models.py - contains backbone vgg and wrapper class to general cascade of conv layer 
   - ssd.py - contains vgg16 + SSD model 
   - ssd_feature_fused_deconv.py - contains SSD model with feature fusion concat module implemented
   - ssd_feature_fused_deconv_eltsum.py - contains SSD model with feature fusion eltsum module implemented 
- data/ -
  - init.py - contains instances:
    - function detection_collate - stack images in 0th dimension and list of tensors with annotations for image and return in tuple format, given tuple of tensor images and list of annotations
    - function base_transform - resize and mean-normalize image
    - class BaseTransform - call base_transform(image) iteratively
  - config.py - configures VOC dataset with source directory, with SSD parameters
  - voc0712.py - configures VOC dataset with labels , and contains :
    - class VOCAnnotationTransform - store dictionaries of classname:index mappings, with an option to discard difficult instances
    - class VOCDetection - update and store annotation based on input image, with functions to get item, pull item, image, annotation and tensor

- layers/ -
  - functions/ - 
    - init.py - import all files in pwd
    - detection.py - contains instances:
      - class Detect - decode location predictions of bboxes and apply NMS based on confidence values and threshold; restrict to tok_k output predictions to reduce noise in results quality (not actual image noise)
        - function forward - forward propagation to update layers given input location prediction, confidence and prior data from their respective layers
    - prior_box.py - contains instances:
      - class PriorBox - collate and store priorbox coordinates in center-offset form and tie it to each source feature map
        - function init - allocate memory and initialize
        - forward - forward propagation through priorbox layers
  - modules/ -
    - init.py - import all files in pwd
    - l2norm.py 
      - class L2Norm - calculate L2 norm and normalize
        - function init - allocate memory and initialize
        - forward - compute the norm and return
    - multibox_loss.py - contains instances:
      - class MultiBoxLoss - compute targets for confidence and localization and apply HNM; using a loss function that is weighted between the cross entropy loss and a smooth L1 loss (weights were found during cross validation)
        - function init - allocate memory and initialize
        - function forward - forward propagate through multibox layers, given tuple of location and confidence predictions, prior box values and ground truth boxes and labels in tensor format
  - init.py - import all files in pwd
  - box_utils.py - contains instances:
    - function point_form - convert prior box values from center-size format for easy comparison to point form ground truth data
    - function center_size - convert prior box values to center-size format for easy comparison to center-size ground truth data
    - function intersect - compute area of intersection between two given boxes
    - function jaccard - compute jaccard overlap or intersection over union of two boxes
    - function match - match prior box with ground truth box (for all boxes) based on highest jaccard overlap, encode in required format (point-form or center-size), and return matching indices for the given confidence and location predictions
    - function encode - encode variances from prior box layers into ground truth boxes
    - function decode - decode locations from priors and locations and return bbox predictions
    - function log_sum_exp - compute log of sum of exponent of difference between current tensor and maximum value of tensor, for unaveraged confidence loss
    - function nms - compute non-maximum suppression to avoid too many overlapping bboxes that highlight nearly the same area
- utils/ -

  - init.py - import all in pwd
  - augmentations.py - contains instances:
    - function intersect - return intersection of two given bounding boxes
    - function jaccard_numpy - return IoU or jaccard overlap of two given bounding boxes
    - class Compose - definitions of different transforms to perform
    - class Lambda - applies a lambda as a transform
    - class ConvertFromInts - convert object from integers
    - class SubtractMeans - subtract mean of image from passed image for normalization
    - class ToAbsoluteCoords - convert lengths (widths, heights) to absolute coordinates
    - class ToPercentCoords - convert coordinates to percentage values of image height and width
    - class Resize - resize image
    - class RandomSaturation - randomly saturate an image
    - class RandomHue - add a random hue to an image
    - class RandomLightingNoise - add random lighting noise to an image
    - class ConvertColor - convert colorspace from BGR to HSV or vice versa
    - class RandomContrast - add random contrast to an image
    - class RandomBrightness - add random brightness to an image
    - class ToCV2Image - shift image to CPU
    - class ToTensor - shift image to GPU
    - class RandomSampleCrop - randomly crop an image and return cropped image, adjusted bounding boxes and new class labels
    - class Expand - expand an image through zero padding and mean-filling, and return along with adjusted bounding boxes and new class labels
    - class RandomMirror - randomly choose to mirror an image
    - class SwapChannels - Transform image by swapping channels in the specified order
    - class PhotometricDistort - apply random brightness and lighting noise, and randomly distort images
    - class SSDAugmentation - itemize all the above transformation functions on every image iteratively
- weights/ - \*.pth files containing pretrained weights of SSD  
- trained_weights/ - \*.pth files having trained weights of experiments for VOC
- requirements.txt - package and module requirements for running the project
- config_vgg16_ssd.py - has settings to run experiments.
- test_images/ - contains imags to test the experimented trained models on web and real world images 

### References 

- [Project Statement for Object-Detection](https://www.charles-deledalle.fr/pages/files/ucsd_ece285_mlip/projectC_object_detection.pdf)
- https://github.com/amdegroot/ssd.pytorch for SSD 
- [INCEPTION SINGLE SHOT MULTIBOX DETECTOR FOR OBJECT DETECTION ](https://sci-hub.tw/10.1109/icmew.2017.8026312)
- https://arxiv.org/ftp/arxiv/papers/1709/1709.05054.pdf
- 


## Contributers - 

- Payam Khorramshahi
- Sanika Patange
- Shivani Athavale 
- Vedavyas Potnuru
