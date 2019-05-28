# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:03:52 2019

@author: Administrator
"""

class DefaultConfig(object):
    num_epochs=60 #设置epoch       *
    epoch_start_i=0
    checkpoint_step=5 #无用
    validation_step=1 #每训练几个epoch进行一次验证
    crop_height=256#无用
    crop_width=256 #无用 
    batch_size=4 #无用            *
    input_channel=1 #输入的图像通道    *
    
    data='/home/FENGsl/Project_template/Dataset'#数据存放的根目录       *
    dataset="Linear_lesion"#数据库名字(需修改成自己的数据名字)          *
    log_dirs='/home/FENGsl/Project_template/Log/Linear_lesion'#存放tensorboard log的文件夹()            *

    lr=0.01    #学习率     *
    lr_mode= 'poly' # poly优化策略
    net_work= 'UNet' # 可选网络      *
    momentum = 0.9# 优化器动量
    weight_decay = 1e-4# L2正则化系数


    mode='train' # 训练模式      *
    k_fold=4 #交叉验证折数        *
    test_fold=4 #测试时候需要选择的第几个文件夹
    num_workers=4 
    num_classes=1 #分割类别数，二类分割设置为1，多类分割设置成 类别数+加背景          **
    cuda='3' #GPU id选择        **
    use_gpu=True 
    pretrained_model_path='/home/FENGsl/Project_template/Linear_lesion_Code/UNet/checkpoints/4/model_057_0.5774.pth.tar' #test的时候模型文件的选择（当mode='test'的时候用）
    save_model_path='./checkpoints'#保存模型的文件夹
    


