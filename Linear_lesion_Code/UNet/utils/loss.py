import torch
import torch.nn as nn
#import torch.nn.functional as F
from torch.autograd import Variable
#
#def loss_builder(loss_type):
#    
#    if loss_type == 'cross_entropy':
#        weight_1 = torch.Tensor([1,5,10,20])
#        criterion = nn.NLLLoss(weight=weight_1,ignore_index=255)
#        criterion_2 = DiceLoss()
#        criterion_3 = nn.BCELoss()
#        return 
#    elif loss_type == 'dice_loss':
#        weight_1 = torch.Tensor([1,5,10,20])
#        criterion_1 = nn.NLLLoss(weight=weight_1,ignore_index=255)
#        criterion_2 = EL_DiceLoss()
#        criterion_3 = nn.BCELoss()
#
#    if loss_type in ['mix_3','mix_33']:
#        criterion_1.cuda()
#        criterion_2.cuda()
#        criterion_3.cuda()
#        criterion = [criterion_1,criterion_2,criterion_3]
#
#    return criterion

class DiceLoss(nn.Module):
    def __init__(self,smooth=0.01):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self,input, target):
        input = torch.sigmoid(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        intersect=(input*target).sum()
        union = torch.sum(input) + torch.sum(target)
        Dice=(2*intersect+self.smooth)/(union+self.smooth)
        dice_loss=1-Dice
        return dice_loss

class Multi_DiceLoss(nn.Module):
    def __init__(self, class_num=4,smooth=0.001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self,input, target):
        input = torch.exp(input)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(0,self.class_num):
            input_i = input[:,i,:,:]
            target_i = (target == i).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice/(self.class_num)
        return dice_loss

class EL_DiceLoss(nn.Module):
    def __init__(self, class_num=4,smooth=1,gamma=0.5):
        super(EL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num
        self.gamma = gamma

    def forward(self,input, target):
        input = torch.exp(input)
        self.smooth = 0.
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        for i in range(1,self.class_num):
            input_i = input[:,i,:,:]
            target_i = (target == i).float()
            intersect = (input_i*target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            if target_i.sum() == 0:
                dice = Variable(torch.Tensor([1]).float()).cuda()
            else:
                dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += (-torch.log(dice))**self.gamma
        dice_loss = Dice/(self.class_num - 1)
        return dice_loss
