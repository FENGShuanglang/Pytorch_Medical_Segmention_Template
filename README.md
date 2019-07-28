# Pytorch_Medical_Segmention_Template

## Introduction
This project is a medical image segmentation template based on ***Pytorch*** implementation, which implements the basic and even most of the functions you need in medical image segmentation experiments. Such as data processing, the design of loss, tool files, save and visualization of log, model files, training ,validation, test and project configuration.

## Folder
- `Dataset`: the folder where dataset is placed.
- `Linear_lesion_Code`: the folder where model and model environment code are placed,`Linear_lesion` is the name of task. Many different models can be put in this folder, for example, I only put `UNet`.
   - `dataset`: the file of data preprocessing.
   - `model`: model files.
   - `utils`: utils files(include many utils)
      - `ddd`
- `Pretrain_model`:  pretriand encoder model,for example,resnet34.

## Prerequisites
- PyTorch 1.0   
   - `conda install torch torchvision`
- tqdm
   - `conda install tqdm`
- imgaug
   - `conda install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely`
   - `conda install imgaug`


## Supported functions:
1. ***Unet-based*** single class segmentation. (Unet parameters are variable, you can adjust the channel reduction factor (feature_scale) according to your actual needs)
2. Automatically realize ***N-fold*** cross-validation
3. Employs ***Dice+BCE*** as a loss function
4. The optimizer is SGD, and the learning strategy is ***'ploy'***
5. Evaluation indicators: ***Acc, Dice, Jaccard, Sen, Spe***
6. Automatically save the N-fold ***checkpoint file***
7. Automatically save the N-fold ***tensorboard log***. Support ***visual comparison*** of multiple experiments before and after, just copy the UNet folder and rename it to: "UNet_xxxxx", then modify it on this.


## What you should do:
## Step 1：
Create a fixed-format data folder under the Dataset folder (using the *Linear_lesion* data as an example, f1, f2, f3.. is the folder name, which stores each fold(N-flod) image):

    ─Linear_lesion
      |
      |─img
      │  ├─f1
      │  ├─f2
      │  ├─f3
      │  ├─f4
      │  └─f5
      └─mask
         ├─f1
         ├─f2
         ├─f3
         ├─f4
         └─f5 
## Step 2：
Modify `Pytorch_Project_template\Linear_lesion_Code\UNet\utils\config.py` according to your needs(rows marked with '\*' require special attention)
## Step 3：
Run the `train.py` file of `Pytorch_Project_template\Linear_lesion_Code\UNet` 
