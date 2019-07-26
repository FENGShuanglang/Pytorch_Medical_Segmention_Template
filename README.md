# Pytorch_Medical_Segmention_Template
## This project implements the following functions:
### 1. ***Unet-based*** single class segmentation. (Unet parameters are variable, you can adjust the channel reduction factor (feature_scale) according to your actual needs)
2. Automatically realize ***N-fold*** cross-validation
3. Employs ***Dice+BCE*** as a loss function
4. The optimizer is SGD, and the learning strategy is ***'ploy'***
5. Evaluation indicators: ***Acc, Dice, Jaccard, Sen, Spe***
6. Automatically save the N-fold ***checkpoint file***
7. Automatically save the N-fold ***tensorboard log***. Support ***visual comparison*** of multiple experiments before and after, just copy the UNet folder and rename it to: "UNet_xxxxx", then modify it on this.

## What you should do if you want to use this project:
## Step 1：
Create a fixed-format data folder under the Dataset folder (using the *Linear_lesion* data as an example, f1, f2, f3.. is the folder name, which stores each fold(N-flod) image):

#### ─Linear_lesion
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
Modify ***Pytorch_Project_template\Linear_lesion_Code\UNet\utils\config.py*** according to your needs(rows marked with '\*' require special attention)
## Step 3：
Run the ***train.py*** file of Pytorch_Project_template\Linear_lesion_Code\UNet 
