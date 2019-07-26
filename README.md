# Pytorch_Medical_Segmention_Template
### This project implements the following functions:
1. ***Unet-based*** single class segmentation. (Unet parameters are variable, you can adjust the channel reduction factor (feature_scale) according to your actual needs)
2. Automatically realize ***N-fold*** cross-validation
3. Employs ***Dice+BCE*** as a loss function
4. The optimizer is SGD, and the learning strategy is ***'ploy'***
5. Evaluation indicators: ***Acc, Dice, Jaccard, Sen, Spe***
6. Automatically save the N-fold ***checkpoint file***
7. Automatically save the N-fold ***tensorboard log***. Support ***visual comparison*** of multiple experiments before and after, just copy the UNet folder and rename it to: "UNet_xxxxx", then modify it on this.

### 想用此工程你需要做的：
#### 第一步：
在Dataset文件夹下创建固定格式的数据文件夹（以Linear_lesion数据为例,f1,f2,f3..是文件夹名字，里面存放每折的图片）：

##### ─Linear_lesion
    ├─img
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
### 第二步：
根据自己的情况修改Pytorch_Project_template\Linear_lesion_Code\UNet\utils\config.py文件(带\*号的是要注意修改的)
### 第三步：
运行Pytorch_Project_template\Linear_lesion_Code\UNet的train.py文件  
