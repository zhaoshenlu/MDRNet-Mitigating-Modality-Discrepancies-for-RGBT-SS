from setproctitle import setproctitle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os, argparse, time, sys, stat, shutil
import numpy as np
from util.util import calculate_accuracy,ReplayBuffer,LambdaLR
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.nn as nn
from util.irseg import IRSeg
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from model.model import TFNet, Discriminator
'''
SEED = 5
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
'''
# config
n_class   = 9
data_dir  = './dataset'  ###  your dataset path  ###
loss_w = torch.tensor([1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000]) #class weights calculated according to ENet

def adjust_learning_rate_D(optimizer, epoch):
    lr = 0.001 * (1-(epoch-1)/500.)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_all(optimizer, epoch):
    lr = args.lr_start * (1-(epoch-1)/500.)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        
def train(epo, model, train_loader, optimizer_all, optimizer_D_RGB, optimizer_D_T):
    
    adjust_learning_rate_all(optimizer_all, epo)
    adjust_learning_rate_D(optimizer_D_RGB, epo)
    adjust_learning_rate_D(optimizer_D_T, epo)
    loss_avg1 = 0.
    loss_avg2 = 0.
    loss_avg3 = 0.
    acc_avg  = 0.
    Tensor = torch.cuda.FloatTensor
    start_t = t =time.time()
    model.train()
    cri1 = nn.CrossEntropyLoss(weight = loss_w).cuda()
    cri2 = nn.BCEWithLogitsLoss().cuda()
    cri_ae = nn.MSELoss().cuda()
    criterion_GAN = nn.BCEWithLogitsLoss().cuda()
    cri_feature = nn.MSELoss().cuda()
    for it, sample in enumerate(train_loader):
        for param in D_RGB.parameters():
            param.requires_grad = False

        for param in D_T.parameters():
            param.requires_grad = False
        RGB_images = Variable(sample['image']).cuda()
        T_images = Variable(sample['thermal']).cuda()
        labels = Variable(sample['label']).cuda()
        real_RGB = RGB_images
        real_T = T_images
        valid = Variable(Tensor(np.ones((real_RGB.size(0), *D_RGB.output_shape))), requires_grad=False)   
        fake = Variable(Tensor(np.zeros((real_RGB.size(0), *D_RGB.output_shape))), requires_grad=False)   
        optimizer_all.zero_grad()
        optimizer_D_RGB.zero_grad()
        optimizer_D_T.zero_grad()
        
        logits, fake_T, fake_RGB, T_recon, RGB_recon,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20 = model(RGB_images, T_images)
        loss_seg = cri1(logits, labels)
        loss_generator_T = cri2(D_T(fake_T), valid)
        loss_generator_RGB = cri2(D_RGB(fake_RGB), valid)
        loss_rescon = cri_ae(T_recon, real_T) + cri_ae(RGB_recon, real_RGB)
        loss_feature = cri_feature(x2, x1) + cri_feature(x4, x3) + cri_feature(x6, x5) + cri_feature(x8, x7) + cri_feature(x10, x9) + cri_feature(x12, x11) + cri_feature(x14, x13) + cri_feature(x16, x15) + cri_feature(x18, x17) + cri_feature(x20, x19)
        RGBclone = fake_RGB.clone().detach()
        Tclone = fake_T.clone().detach()
        loss_m = loss_seg + 0.01*loss_generator_T + 0.01*loss_generator_RGB + 0.1*loss_rescon + loss_feature
        loss_m.backward()
        optimizer_all.step()
        
        for param in D_RGB.parameters():
            param.requires_grad = True

        for param in D_T.parameters():
            param.requires_grad = True
            
        optimizer_D_RGB.zero_grad()
        optimizer_D_T.zero_grad()
        loss_real_RGB = criterion_GAN(D_RGB(real_RGB), valid)
        #fake_RGB_ = fake_RGB_buffer.push_and_pop(RGBclone)                  
        loss_fake_RGB = criterion_GAN(D_RGB(RGBclone), fake)          
        loss_D_RGB = (loss_real_RGB + loss_fake_RGB) / 2
        loss_D_RGB.backward()
        optimizer_D_RGB.step()
        
        loss_real_T = criterion_GAN(D_T(real_T), valid)
        #fake_T_ = fake_T_buffer.push_and_pop(Tclone)                    
        loss_fake_T = criterion_GAN(D_T(Tclone), fake)          
        loss_D_T = (loss_real_T + loss_fake_T) / 2
        loss_D_T.backward()
        optimizer_D_T.step()
        
        acc = calculate_accuracy(logits, labels)
        loss_avg1 += float(loss_m)
        loss_avg2 += float(loss_D_RGB)
        loss_avg3 += float(loss_D_T)
        acc_avg  += float(acc)
        cur_t = time.time()
        if cur_t-t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss_seg: %.4f loss_generator_T: %.4f loss_generator_RGB: %.4f loss_rescon: %.4f loss_feature: %.4f loss_D_RGB: %.4f loss_D_T: %.4f acc: %.4f' \
                % (epo, args.epoch_max, it+1, train_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss_seg), float(0.01*loss_generator_T), float(0.01*loss_generator_RGB), float(loss_rescon*0.1), float(loss_feature), float(loss_D_RGB), float(loss_D_T), float(acc)))
            t += 5

    content = '| epo:%s/%s train_loss_m_avg:%.4f train_loss_D_RGB_avg:%.4f train_loss_D_T_avg:%.4f train_acc_avg:%.4f ' \
            % (epo, args.epoch_max, loss_avg1/train_loader.n_iter, loss_avg2/train_loader.n_iter, loss_avg3/train_loader.n_iter, acc_avg/train_loader.n_iter)
    print(content)

def testing(epo, model, test_loader):
    model.eval()
    conf_total = np.zeros((n_class, n_class))

    with torch.no_grad():
        for it, sample in enumerate(test_loader):
            RGB_images = Variable(sample['image']).cuda()
            T_images = Variable(sample['thermal']).cuda()
            labels = Variable(sample['label']).cuda()
            logits, fake_T, fake_RGB, T_recon, RGB_recon,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20 = model(RGB_images, T_images)
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten() # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(label, prediction, labels = [0,1,2,3,4,5,6,7,8]) # conf is n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf

            print('|- %s, epo %s/%s. testing iter %s/%s.' % (args.model_name, epo, args.epoch_max, it+1, test_loader.n_iter))

    precision, recall, IoU, = compute_results(conf_total)
    
    return np.mean(np.nan_to_num(precision)), np.mean(np.nan_to_num(recall)), np.mean(np.nan_to_num(IoU))

def main():

    train_dataset = IRSeg(mode='train', do_aug=True)
    test_dataset = IRSeg(mode='test', do_aug=False)
    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    test_loader = DataLoader(
        dataset      = test_dataset,
        batch_size   = 1,
        shuffle      = False,
        num_workers  = args.num_workers,
        pin_memory   = True,
        drop_last    = False
    )
    train_loader.n_iter = len(train_loader)
    test_loader.n_iter = len(test_loader)
    max_R = 0
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' %(args.model_name, epo))

        train(epo, model, train_loader, optimizer_all, optimizer_D_RGB, optimizer_D_T)
        P, R, I = testing(epo, model, test_loader)
        checkpoint_model_file1 = os.path.join(r'', str(epo)+'.pth')####  your save path  ####
        torch.save(model.state_dict(), checkpoint_model_file1)
        print('OK!')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train with pytorch')
    ############################################################################################# 
    parser.add_argument('--model_name',  '-M',  type=str, default='TFNet')  ####  MDRNet+  ####
    parser.add_argument('--batch_size',  '-B',  type=int, default=2) 
    parser.add_argument('--lr_start',  '-LS',  type=float, default=0.001)
    parser.add_argument('--lr_decay', '-LD', type=float, default=0.95)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=501) # please stop training mannully
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1) 
    parser.add_argument('--num_workers', '-j',  type=int, default=8)
    args = parser.parse_args()
                 
    model = eval(args.model_name)(n_class=n_class)    
    model.cuda()
    input_shape_1 = (3, 480, 640)
    input_shape_2 = (3, 480, 640)
    D_RGB = Discriminator(input_shape_1).cuda()
    D_T = Discriminator(input_shape_2).cuda()
    optimizer_all = torch.optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    optimizer_D_RGB = torch.optim.Adam(D_RGB.parameters(), lr=0.001, betas=(0.5, 0.999))
    optimizer_D_T = torch.optim.Adam(D_T.parameters(), lr=0.001, betas=(0.5, 0.999))
    print('training %s on GPU with pytorch' % (args.model_name))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))

    main()
