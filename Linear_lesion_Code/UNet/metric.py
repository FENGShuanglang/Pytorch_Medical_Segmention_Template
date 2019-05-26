# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 10:59:53 2018

"""

import numpy as np
import os 
from PIL import Image
#path_true=r'D:\task\projects\cabunet\keras_cabunet\aug\data\testset\label\4'
#path_predict=r'D:\task\projects\cabunet\keras_cabunet\aug\result3\2\4'
path_true=r'G:\KeTi\JBHI_pytorch\BaseNet\dataset\path\to\PiFu\mask\f1'
path_predict=r'G:\KeTi\JBHI_pytorch\BaseNet_Full_channel_impore\dataset\test\f1'
TP=FPN=0
Jaccard=[]
for roots,dirs,files in os.walk(path_predict):
    if files:
#        dice=[]
#        num=0
        for file in files:
#            num=num+1
            pre_file_path=os.path.join(roots,file)
            true_file_path=os.path.join(path_true,file)
            img_pre = np.array(Image.open(pre_file_path).convert("L"))
            img_pre[img_pre==255]=1
            img_true = np.array(Image.open(true_file_path).convert("L"))
            img_true[img_true==255]=1
#            print(img_pre.shape)
#            print(img_true.shape)
#            TP = TP+np.sum(np.array(img_pre,dtype=np.int32)&np.array(img_true,dtype=np.int32))
#            FPN = FPN +np.sum(np.array(img_pre,dtype=np.int32)|np.array(img_true,dtype=np.int32))
            TP = TP+np.sum(img_pre*img_true)
            FPN = FPN +np.sum(img_pre)+np.sum(img_true)
            single_I=np.sum(img_pre*img_true)
            single_U=np.sum(img_pre)+np.sum(img_true)-single_I
            Jaccard.append(single_I/single_U)



            
dice = 2*TP/FPN
print('TP:',TP)
print('FPN:',FPN)           
print("DICE",dice)
print('glob_Jaccard',TP/(FPN-TP))
print('single_Jaccard',sum(Jaccard)/len(Jaccard))
            
            
#            
##            pre_npy=np.load(pre_file_path)
##            true_npy=np.load(true_file_path)
#            for i in range(1,4):
#                pre_npy_s=np.zeros(pre_npy.shape)
#                true_npy_s=np.zeros(true_npy.shape)
#                
#                pre_npy_s[pre_npy==i]=1 
#                true_npy_s[true_npy==i]=1
#                TP=np.sum(np.array(pre_npy_s,dtype=np.int32)&np.array(true_npy_s,dtype=np.int32))
#                FPN=np.sum(np.array(pre_npy_s,dtype=np.int32)|np.array(true_npy_s,dtype=np.int32))
#                print('%d_TP:' %num,TP)
#                print('%d_FPN:' %num,FPN)
##                if FPN!=0:
#                    
#                if np.sum(true_npy_s)!=0:
#                    dice_cof=2*TP/(TP+FPN)
#                    dice.append(dice_cof)
#dice=np.array(dice)
#print(dice)
#dice_mean=np.mean(dice)
#print(dice_mean)