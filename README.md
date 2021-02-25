# Pathofusion
A deep learning based framework for recognition and fusion of multimodal histopathological images

### Good news! The source for Labelling website is now released, please see the folder "LabelingWebsite" or visit standalone code base: https://github.com/guoqingbao/Patholabelling

#### Detecting malignant features from pathology images (visualized as heatmaps)
![](/others/prediction.gif)

## Prerequisites
The following python libraries are required:

matplotlib, sqlite3, pandas, scipy, scikit-learn, pytorch, tensorflow and keras

#### Architecture
![](/others/architecture.png)

#### CAM visualization illustrates the underlying mechanism of the bifocal design
![](/others/cam.png)


## The dataset
The datasets used in our study were provided under folder "data". Raw data may be provided upon requests. 


## Pretrained models
Pretrained models were provided under results/bcnn (torch_model.h5 and torch_model_cd276.h5) for recognition of two modality neuro-features from whole-slide images. Please refer to BrainPredction.py or BrainPredction-276.py. Colour normalization of your whole-slides images is required before using the pretrained models (a reference image is given under folder "others"). You may also train your own model using code provided. Usually, 30+ training cases are recommended.

## The marking system (labelling website)
We also provide the source code for the pathology image labelling website, and you can perform off-line marking of your own whole-slides images in the intranet. You can use code provided (ExtractImagePatches.py) to establish your pathology database first, then you train your model using code provided. For non-commercial usage, please contact corresponding author to obtain the source code of the labelling website.
#### Please see the folder "LabelingWebsite", newest updates can be found in the standalone code base: https://github.com/guoqingbao/Patholabelling


## Video demonstration

https://cloudstor.aarnet.edu.au/plus/s/dVmEp2R87lFhc6v

In the video, the original H&E is first shown; 
next, the predicted heatmap was overlaid; 
fianlly, the prediction was compared with expert markings.

# The code

### 1. Extracting image patches (ExtractImagePatches.py)
Make sure you deployed the labelling website first. After you finished image marking, you use this module to extract paired patches from the whole-slide images using marking coordinates saved in MySql database (website database). While you can also use our provided datasets (under folder "data") for reproducibility measurements.

### 2. Model structure
Please refer to folder "models" and BrainModel.py.

### 3. Training models for recognition
You can use BrainModel.py and BrainModel-CD276.py to train models for the recognition. Please make sure you have downloaded datasets to folder "data" before running the code. 

The training and test perfromance metrics of different models were recorded in the folder "results".

### 4. Model performance comparison
The model introduced in this study was compared with Xception (BrainXception.py), Xception (transfer learning part in BrainXception.py) and a subnet CNN (Single-input ordinary model, BrainModel-SubNet.py). 

### 5. Predicting heatmaps
After the models trained (or using trained models provided), you can predict the heatmaps of unknown histological slides using BrainPredction.py and BrainPredction-CD276.py. 

### 6. The fusion
BrainPredction.py provides functions for fusion of multimodal whole-slide images. 

### 7. Multimodal overlap analysis
Please refer to BrainPositivePercent.py.

