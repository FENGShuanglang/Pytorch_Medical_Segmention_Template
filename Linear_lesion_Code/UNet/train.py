import argparse
#from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.Linear_lesion import LinearLesion
import socket
from datetime import datetime

import os
from model.unet import UNet
import torch
from tensorboardX import SummaryWriter
import tqdm
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from  PIL import Image
#from utils import poly_lr_scheduler
#from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy,batch_intersection_union,batch_pix_accuracy
import utils.utils as u
import utils.loss as LS
from utils.config import DefaultConfig
import torch.backends.cudnn as cudnn
def val(args, model, dataloader):
    print('\n')
    print('Start Validation!')
    with torch.no_grad():
        model.eval()
        tbar = tqdm.tqdm(dataloader, desc='\r')
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0

        total_Dice=[]
        total_Acc=[]
        total_jaccard=[]
        total_Sensitivity=[]
        total_Specificity=[]

        for i, (data, label) in enumerate(tbar):
            # tbar.update()
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            
            # get RGB predict image
            aux_predict,predict = model(data)
            Dice, Acc, jaccard, Sensitivity, Specificity=u.eval_single_seg(predict,label)

            total_Dice+=Dice
            total_Acc+=Acc
            total_jaccard+=jaccard
            total_Sensitivity+=Sensitivity
            total_Specificity+=Specificity

            dice=sum(total_Dice) / len(total_Dice)
            acc=sum(total_Acc) / len(total_Acc)
            jac=sum(total_jaccard) / len(total_jaccard)
            sen=sum(total_Sensitivity) / len(total_Sensitivity)
            spe=sum(total_Specificity) / len(total_Specificity)

            tbar.set_description(
                'Dice: %.3f, Acc: %.3f, Jac: %.3f, Sen: %.3f, Spe: %.3f' % (dice,acc,jac,sen,spe))


        print('Dice:',dice)
        print('Acc:',acc)
        print('Jac:',jac)
        print('Sen:',sen)
        print('Spe:',spe)
        return dice,acc,jac,sen,spe
    


def train(args, model, optimizer,criterion, dataloader_train, dataloader_val,writer,k_fold):
    
    step = 0
    best_pred=0.0
    for epoch in range(args.num_epochs):
        lr = u.adjust_learning_rate(args,optimizer,epoch) 
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('fold %d,epoch %d, lr %f' % (int(k_fold),epoch, lr))
        loss_record = []
        train_loss=0.0
#        is_best=False
        for i,(data, label) in enumerate(dataloader_train):
            # if i>len(dataloader_train)-2:
            #     break
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            aux_out,main_out = model(data)
            # get weight_map
            weight_map=torch.zeros(args.num_classes)
            weight_map=weight_map.cuda()
            for t in range(args.num_classes):
                weight_map[t]=1/(torch.sum((label==t).float())+1.0)
            # print(weight_map)

            loss_aux=F.binary_cross_entropy_with_logits(main_out,label,weight=None)
            loss_main= criterion[1](main_out, label)

            loss =loss_main+loss_aux
            loss.backward()
            optimizer.step()
            tq.update(args.batch_size)
            train_loss += loss.item()
            tq.set_postfix(loss='%.6f' % (train_loss/(i+1)))
            step += 1
            if step%10==0:
                writer.add_scalar('Train/loss_step_{}'.format(int(k_fold)), loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('Train/loss_epoch_{}'.format(int(k_fold)), float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        

        if epoch % args.validation_step == 0:
            Dice, Acc, jaccard, Sensitivity, Specificity= val(args, model, dataloader_val)
            writer.add_scalar('Valid/Dice_val_{}'.format(int(k_fold)), Dice, epoch)
            writer.add_scalar('Valid/Acc_val_{}'.format(int(k_fold)), Acc, epoch)
            writer.add_scalar('Valid/Jac_val_{}'.format(int(k_fold)), jaccard, epoch)
            writer.add_scalar('Valid/Sen_val_{}'.format(int(k_fold)), Sensitivity, epoch)
            writer.add_scalar('Valid/Spe_val_{}'.format(int(k_fold)), Specificity, epoch)

            is_best=Dice > best_pred
            best_pred = max(best_pred, Dice)
            checkpoint_dir_root = args.save_model_path
            checkpoint_dir=os.path.join(checkpoint_dir_root,str(k_fold))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            checkpoint_latest =os.path.join(checkpoint_dir, 'checkpoint_latest.pth.tar')
            u.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_dice': best_pred,
                    }, best_pred,epoch,is_best, checkpoint_dir,filename=checkpoint_latest)
                    
def eval(model,dataloader, args):
    print('start test!')
    with torch.no_grad():
        model.eval()
        # precision_record = []
        tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
        tq.set_description('test')
        # total_dice,total_precision,total_jaccard=0,0,0
        comments=os.getcwd().split('/')[-1]
        for i, (data, label_path) in enumerate(dataloader):
            tq.update(args.batch_size)
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                # label = label.cuda()
            aux_pred,predict = model(data)
            predict=torch.round(torch.sigmoid(aux_pred)).byte()
            # Dice,Precsion,Jaccard= u.eval_seg(predict, label)
            # total_dice+=Dice
            # total_precision+=Precsion
            # total_jaccard+=Jaccard
            pred_seg=predict.data.cpu().numpy()*255
            
            for index,item in enumerate(label_path):
                save_img_path=label_path[index].replace('mask',comments+'_mask')
                if not os.path.exists(os.path.dirname(save_img_path)):
                    os.makedirs(os.path.dirname(save_img_path))
                img=Image.fromarray(pred_seg[index].squeeze(),mode='L')
                img.save(save_img_path)
                tq.set_postfix(str=str(save_img_path))
        tq.close()
            
def main(mode='train',args=None,writer=None,k_fold=1):


    # create dataset and dataloader
    dataset_path = os.path.join(args.data, args.dataset)
    dataset_train = LinearLesion(dataset_path, scale=(args.crop_height, args.crop_width),k_fold_test=k_fold,mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )
    
    dataset_val = LinearLesion(dataset_path, scale=(args.crop_height, args.crop_width),k_fold_test=k_fold,mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=len(args.cuda.split(',')),# the default is 1(the number of gpu), you can set it to what you want
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )

    dataset_test = LinearLesion(dataset_path, scale=(args.crop_height, args.crop_width),k_fold_test=k_fold,mode='test')
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=len(args.cuda.split(',')),# the default is 1(the number of gpu), you can set it to what you want
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True 
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    
    
    
    #load model
    model_all={'UNet':UNet(in_channels=args.in_channels, n_classes=args.num_classes)}
    model=model_all[args.net_work]
    cudnn.benchmark = True
    # model._initialize_weights()
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    # load pretrained model if exists
    if args.pretrained_model_path and mode=='test':
        print("=> loading pretrained model '{}'".format(args.pretrained_model_path))
        checkpoint = torch.load(args.pretrained_model_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Done!')
        
        

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # criterion_aux=nn.NLLLoss(weight=None)
    criterion_aux=nn.BCEWithLogitsLoss(weight=None)
    criterion_main=LS.DiceLoss()
    criterion=[criterion_aux,criterion_main]
    if mode=='train':
        train(args, model, optimizer,criterion, dataloader_train, dataloader_val,writer,k_fold)
    if mode=='test':
        eval(model,dataloader_test, args)
    if mode=='train_test':
        train(args, model, optimizer,criterion, dataloader_train, dataloader_val)
        eval(model,dataloader_test, args)




if __name__ == '__main__':
    seed=1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    args=DefaultConfig()
    modes=args.mode

    if modes=='train':
        comments=os.getcwd().split('/')[-1]
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(args.log_dirs, comments+'_'+current_time + '_' + socket.gethostname())
        writer = SummaryWriter(log_dir=log_dir)
        for i in range(args.k_fold):
            main(mode='train',args=args,writer=writer,k_fold=int(i+1))
    elif modes=='test':
         main(mode='test',args=args,writer=None,k_fold=args.test_fold)

