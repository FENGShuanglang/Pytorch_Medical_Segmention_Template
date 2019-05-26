import torch
import glob
import os
from torchvision import transforms
from torchvision.transforms import functional as F
#import cv2
from PIL import Image
# import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
#from utils import get_label_info, one_hot_it
import random

def augmentation():
    # augment images with spatial transformation: Flip, Affine, Rotation, etc...
    # see https://github.com/aleju/imgaug for more details
    pass

def augmentation_pixel():
    # augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
    pass

class LinearLesion(torch.utils.data.Dataset):
    def __init__(self, dataset_path,scale,k_fold_test=1, mode='train'):
        super().__init__()
        self.mode = mode
        self.img_path=dataset_path+'/img'
        self.mask_path=dataset_path+'/mask'
        self.image_lists,self.label_lists=self.read_list(self.img_path,k_fold_test=k_fold_test)
        self.flip =iaa.SomeOf((2,4),[
             iaa.Fliplr(0.5),
             iaa.Flipud(0.5),
             iaa.Affine(rotate=(-30, 30)),
             iaa.AdditiveGaussianNoise(scale=(0.0,0.08*255))], random_order=True)
        # resize
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        # normalization
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        # load image and crop
        img = Image.open(self.image_lists[index])

        
        img = np.array(img)
        labels=self.label_lists[index]

        #load label
        if self.mode !='test':
            label = Image.open(self.label_lists[index])
          
            label = np.array(label) 
            label[label!=127]=0
            label[label==127]=1
            # label=np.argmax(label,axis=-1)
            # label[label!=1]=0
            # augment image and label
            if self.mode == 'train':
                
                
                seq_det = self.flip.to_deterministic()#
                segmap = ia.SegmentationMapOnImage(label, shape=label.shape, nb_classes=2)

                img = seq_det.augment_image(img)
                label = seq_det.augment_segmentation_maps([segmap])[0].get_arr_int().astype(np.uint8)

            label=np.reshape(label,(1,)+label.shape)
            label=torch.from_numpy(label.copy()).float()
           

            labels=label
           
        # img=np.reshape(img,img.shape+(1,))       # 如果输入是1通道需打开此注释 ******
        img = self.to_tensor(img.copy()).float()
        return img, labels

    def __len__(self):
        return len(self.image_lists)
    def read_list(self,image_path,k_fold_test=1):
        fold=sorted(os.listdir(image_path))
        # print(fold)
        os.listdir()
        img_list=[]
        if self.mode=='train':
            fold_r=fold
            fold_r.remove('f'+str(k_fold_test))# remove testdata
            for item in fold_r:
                img_list+=glob.glob(os.path.join(image_path,item)+'/*.png')
            # print(len(img_list))
            label_list=[x.replace('img','mask').split('.')[0]+'.png' for x in img_list]
        elif self.mode=='val' or self.mode=='test':
            fold_s=fold[k_fold_test-1]
            img_list=glob.glob(os.path.join(image_path,fold_s)+'/*.png')
            label_list=[x.replace('img','mask').split('.')[0]+'.png' for x in img_list]
        return img_list,label_list



                





#if __name__ == '__main__':
#    data = LinearLesion(r'K:\Linear Lesion\Linear_lesion', (512, 512),mode='train')
#    # from model.build_BiSeNet import BiSeNet
#    # from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy
#    from torch.utils.data import DataLoader
#    dataloader_test = DataLoader(
#        data,
#        # this has to be 1
#        batch_size=4,
#        shuffle=True,
#        num_workers=0,
#        pin_memory=True,
#        drop_last=True 
#    )
#    for i, (img, label) in enumerate(dataloader_test):
#
#        print(label)
#        print(img)
#        if i>3:
#            break

   

