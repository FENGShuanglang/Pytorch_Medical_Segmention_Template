#import torch.nn as nn
import torch
from torch.nn import functional as F
#from PIL import Image
import numpy as np
import pandas as pd
#import os
import os.path as osp
import shutil
#import math

def save_checkpoint(state,best_pred, epoch,is_best,checkpoint_path,filename='./checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, osp.join(checkpoint_path,'model_{:03d}_{:.4f}.pth.tar'.format((epoch + 1),best_pred)))



def adjust_learning_rate(opt, optimizer, epoch):                        
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr




def one_hot_it(label, label_info):
	# return semantic_map -> [H, W, num_classes]
	semantic_map = []
	for info in label_info:
		color = label_info[info]
		# colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
		equality = np.equal(label, color)
		class_map = np.all(equality, axis=-1)
		semantic_map.append(class_map)
	semantic_map = np.stack(semantic_map, axis=-1)
	return semantic_map
    
    
def compute_score_multi(predict, target, forground = 1,smooth=1):
    score = 0
    count = 0
    target[target!=forground]=0
    predict[predict!=forground]=0
    assert(predict.shape == target.shape)
    overlap = ((predict == forground)*(target == forground)).sum() #TP
    union=(predict == forground).sum() + (target == forground).sum()-overlap #FP+FN+TP
    FP=(predict == forground).sum()-overlap #FP
    FN=(target == forground).sum()-overlap #FN
    TN= target.shape[0]*target.shape[1]-union #TN


    #print('overlap:',overlap)
    dice=(2*overlap +smooth)/ (union+overlap+smooth)
    
    precsion=((predict == target).sum()+smooth) / (target.shape[0]*target.shape[1]+smooth)
    
    jaccard=(overlap+smooth) / (union+smooth)

    Sensitivity=(overlap+smooth) / ((target == forground).sum()+smooth)

    Specificity=(TN+smooth) / (FP+TN+smooth)
    

    return dice,precsion,jaccard,Sensitivity,Specificity



def eval_multi_seg(predict, target, forground = 1):
    pred_seg=torch.argmax(torch.exp(predict),dim=1).int()
    pred_seg = pred_seg.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert(pred_seg.shape == label_seg.shape)

    Dice = []
    Precsion = []
    Jaccard = []
    Sensitivity=[]
    Specificity=[]

    n = pred_seg.shape[0]
    
    for i in range(n):
        dice,precsion,jaccard,sensitivity,specificity= compute_score_multi(pred_seg[i],label_seg[i])
        Dice.append(dice)
        Precsion .append(precsion)
        Jaccard.append(jaccard)
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)

    return Dice,Precsion,Jaccard,Sensitivity,Specificity

def compute_score_single(predict, target, forground = 1,smooth=1):
    score = 0
    count = 0
    target[target!=forground]=0
    predict[predict!=forground]=0
    assert(predict.shape == target.shape)
    overlap = ((predict == forground)*(target == forground)).sum() #TP
    union=(predict == forground).sum() + (target == forground).sum()-overlap #FP+FN+TP
    FP=(predict == forground).sum()-overlap #FP
    FN=(target == forground).sum()-overlap #FN
    TN= target.shape[0]*target.shape[1]*target.shape[2]-union #TN


    #print('overlap:',overlap)
    dice=(2*overlap +smooth)/ (union+overlap+smooth)
    
    precsion=((predict == target).sum()+smooth) / (target.shape[0]*target.shape[1]*target.shape[2]+smooth)
    
    jaccard=(overlap+smooth) / (union+smooth)

    Sensitivity=(overlap+smooth) / ((target == forground).sum()+smooth)

    Specificity=(TN+smooth) / (FP+TN+smooth)
    

    return dice,precsion,jaccard,Sensitivity,Specificity

def eval_single_seg(predict, target, forground = 1):
    pred_seg=torch.round(torch.sigmoid(predict)).int()
    pred_seg = pred_seg.data.cpu().numpy()
    label_seg = target.data.cpu().numpy().astype(dtype=np.int)
    assert(pred_seg.shape == label_seg.shape)

    Dice = []
    Precsion = []
    Jaccard = []
    Sensitivity=[]
    Specificity=[]

    n = pred_seg.shape[0]
    
    for i in range(n):
        dice,precsion,jaccard,sensitivity,specificity= compute_score_single(pred_seg[i],label_seg[i])
        Dice.append(dice)
        Precsion .append(precsion)
        Jaccard.append(jaccard)
        Sensitivity.append(sensitivity)
        Specificity.append(specificity)

    return Dice,Precsion,Jaccard,Sensitivity,Specificity
    

def batch_pix_accuracy(pred,label,nclass=1):
    if nclass==1:
        pred=torch.round(torch.sigmoid(pred)).int()
        pred=pred.cpu().numpy()
    else:
        pred=torch.max(pred,dim=1)
        pred=pred.cpu().numpy()
    label=label.cpu().numpy()
    pixel_labeled = np.sum(label >=0)
    pixel_correct=np.sum(pred==label)
    
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    
    return pixel_correct,pixel_labeled

def batch_intersection_union(predict, target, nclass):

    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int),note: not include background
    """
    if nclass==1:
        pred=torch.round(torch.sigmoid(predict)).int()
        pred=pred.cpu().numpy()
        target = target.cpu().numpy()
        area_inter=np.sum(pred*target)
        area_union=np.sum(pred)+np.sum(target)-area_inter

        return area_inter,area_union




    if nclass>1:
        _, predict = torch.max(predict, 1)
        mini = 1
        maxi = nclass
        nbins = nclass
        predict = predict.cpu().numpy() + 1
        target = target.cpu().numpy() + 1
        # target = target + 1
        
        predict = predict * (target > 0).astype(predict.dtype)
        intersection = predict * (predict == target)
        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins-1, range=(mini+1, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins-1, range=(mini+1, maxi))
        area_lab, _ = np.histogram(target, bins=nbins-1, range=(mini+1, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all(), \
        	"Intersection area should be smaller than Union area"
        return area_inter, area_union
    

def pixel_accuracy(im_pred, im_lab):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)

    # Remove classes from unlabeled pixels in gt image. 
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(im_lab > 0)
    pixel_correct = np.sum((im_pred == im_lab) * (im_lab > 0))
    #pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_correct, pixel_labeled

def reverse_one_hot(image):
	"""
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,1])

	# for i in range(0, w):
	#     for j in range(0, h):
	#         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
	#         x[i, j] = index
	image = image.permute(1, 2, 0)
	x = torch.argmax(image, dim=-1)
	return x


def colour_code_segmentation(image, label_values):
	"""
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,3])
	# colour_codes = label_values
	# for i in range(0, w):
	#     for j in range(0, h):
	#         x[i, j, :] = colour_codes[int(image[i, j])]
	label_values = [label_values[key] for key in label_values]
	colour_codes = np.array(label_values)
	x = colour_codes[image.astype(int)]

	return x

#def compute_global_accuracy(pred, label):
#	pred = pred.flatten()
#	label = label.flatten()
#	total = len(label)
#	count = 0.0
#	for i in range(total):
#		if pred[i] == label[i]:
#			count = count + 1.0
#	return float(count) / float(total)