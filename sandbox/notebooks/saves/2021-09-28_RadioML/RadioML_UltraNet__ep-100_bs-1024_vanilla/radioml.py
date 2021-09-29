from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from torch.utils.data.sampler import *
from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import utils.utils as util
import utils.quantization as q
import h5py

from models import *

import numpy as np
import os, time, sys
from datetime import datetime
import copy
import argparse
import random
from sklearn.metrics import accuracy_score

#----------------------------
# Argument parser.
#----------------------------
parser = argparse.ArgumentParser(description='PyTorch RadioML Training')
parser.add_argument('--data_dir', '-d', type=str, default='../../../RadioML/Data/GOLD_XYZ_OSC.0001_1024.hdf5',
                    help='path to the dataset directory')
parser.add_argument('--arch', metavar='ARCH', default='resnet', help='Choose a model')
parser.add_argument('--save', '-s', action='store_true', help='save the model')
parser.add_argument('--save_folder', type=str, default='./saves/',
                    help='Folder to save checkpoints and log.')
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train')
parser.add_argument('--batch_size', '-b', type=int, default=1024, help='Batch size')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--num_classes', type=int, default=24, help='Number of classes')
parser.add_argument('--finetune', '-f', action='store_true', help='finetune the model')
parser.add_argument('--test', '-t', action='store_true', help='test only')
parser.add_argument('--resume', '-r', type=str, default=None,
                    help='path of the model checkpoint for resuming training')

parser.add_argument('--which_gpus', '-gpu', type=str, default='0', help='which gpus to use')
args = parser.parse_args()
print(args)


#----------------------------
# Parameters
#----------------------------
batch_size = args.batch_size
num_epoch = args.epochs
# batch_size = 128
# num_epoch = 250
_LAST_EPOCH = -1 #last_epoch arg is useful for restart
# _WEIGHT_DECAY = 1e-5 # 0.
# candidates = ['binput-prerprelu-pg', 'binput-prerprelu-1bit', 'binput-prerprelu-2bit', 'prerprelu-resnet20']
_ARCH = args.arch
# this_file_path = os.path.dirname(os.path.abspath(__file__))
save_folder = args.save_folder

now = datetime.now()
print("Now is", now)
dt_str = now.strftime("%y%m%d-%H%M")
print("date and time =", dt_str)

torch.manual_seed(0)
np.random.seed(0)


#----------------------------
# Load the dataset
#----------------------------
class radioml_18_dataset(Dataset):
    def __init__(self, dataset_path):
        super(radioml_18_dataset, self).__init__()
        h5_file = h5py.File(dataset_path,'r')
        self.data = h5_file['X']
        self.mod = np.argmax(h5_file['Y'], axis=1) # comes in one-hot encoding
        self.snr = h5_file['Z'][:,0]
        self.len = self.data.shape[0]

        self.mod_classes = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK',
        '16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM',
        'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']
        self.snr_classes = np.arange(-20., 32., 2) # -20dB to 30dB

        # do not touch this seed to ensure the prescribed train/test split!
        np.random.seed(2018)
        train_indices = []
        test_indices = []
        for mod in range(0, 24): # all modulations (0 to 23)
            for snr_idx in range(0, 26): # all SNRs (0 to 25 = -20dB to +30dB)
                # 'X' holds frames strictly ordered by modulation and SNR
                start_idx = 26*4096*mod + 4096*snr_idx
                indices_subclass = list(range(start_idx, start_idx+4096))
                
                # 90%/10% training/test split, applied evenly for each mod-SNR pair
                split = int(np.ceil(0.1 * 4096)) 
                np.random.shuffle(indices_subclass)
                train_indices_subclass = indices_subclass[split:]
                test_indices_subclass = indices_subclass[:split]
                
                # you could train on a subset of the data, e.g. based on the SNR
                # here we use all available training samples
                if snr_idx >= 0:
                    train_indices.extend(train_indices_subclass)
                test_indices.extend(test_indices_subclass)
                
        self.train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        self.test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # transpose frame into Pytorch channels-first format (NCL = -1,2,1024)
        return self.data[idx].transpose(), self.mod[idx], self.snr[idx]

    def __len__(self):
        return self.len


#----------------------------
# Train the network.
#----------------------------
def train_model(trainloader, testloader, net, 
                optimizer, scheduler, start_epoch, device, log):
    # define the loss function
    criterion = (nn.CrossEntropyLoss().cuda() 
                if torch.cuda.is_available() else nn.CrossEntropyLoss())

    best_acc = 0.0
    best_model = copy.deepcopy(net.state_dict())

    start_time = time.time()
    epoch_time = util.AverageMeter('Time/Epoch', ':.2f')

    recorder = util.RecorderMeter(num_epoch)
    
    train_los = 0
    val_los = 0
    train_acc = 0
    val_acc = 0

    for epoch in range(start_epoch, num_epoch): # loop over the dataset multiple times

        # set printing functions
        # batch_time = util.AverageMeter('Time/batch', ':.2f')
        # losses = util.AverageMeter('Loss', ':6.2f')
        # top1 = util.AverageMeter('Acc', ':6.2f')
        # progress = util.ProgressMeter(
        #                 len(trainloader),
        #                 [losses, top1, batch_time],
        #                 prefix="Epoch: [{}]".format(epoch+1)
        #                 )

        # switch the model to the training mode
        net.train()

        current_learning_rate = optimizer.param_groups[0]['lr']
        # print_log('current learning rate = {}'.format(current_learning_rate), log)
        

        need_hour, need_mins, need_secs = util.convert_secs2time(
            epoch_time.avg * (num_epoch - epoch))
        avg_hour, avg_mins, avg_secs = util.convert_secs2time(
            epoch_time.avg)
        need_time = '[Need: {:02d}h{:02d}m]'.format(need_hour, need_mins)
        avg_time = '[Avg: {:02d}m{:02d}s]'.format(avg_mins, avg_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s}{:s} [LR={:6.5f}]'.format(util.time_string(), epoch, num_epoch,
                                                                                   need_time, avg_time, current_learning_rate,)
            + ' [Best Acc={:.2f}]'.format(best_acc), log)

        # each epoch
        batch_time = util.AverageMeter('Time/batch', ':.2f')
        losses = util.AverageMeter('Loss', ':6.2f')
        top1 = util.AverageMeter('Acc', ':6.2f')
        top5 = util.AverageMeter('Acc', ':6.2f')
        
        start = time.time()
        end = time.time()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # # measure accuracy and record loss
            # _, batch_predicted = torch.max(outputs.data, 1)
            # batch_accu = 100.0 * (batch_predicted == labels).sum().item() / labels.size(0)
            # losses.update(loss.item(), labels.size(0))
            # top1.update(batch_accu, labels.size(0))

            # measure accuracy and record loss
            def accuracy(outputs, labels, topk=(1,)):
                """Computes the precision@k for the specified values of k"""
                with torch.no_grad():
                    maxk = max(topk)
                    batch_size = labels.size(0)

                    _, pred = outputs.topk(maxk, 1, True, True)
                    pred = pred.t()
                    correct = pred.eq(labels.view(1, -1).expand_as(pred))

                    res = []
                    for k in topk:
                        correct_k = correct[:k].reshape(-1).float().sum(0)
                        res.append(correct_k.mul_(100.0 / batch_size))
                    return res

            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            elapsed = end - start

            prt_str = ('\r    Epoch: [%d][%d/%d]  Time %.3f  Loss %.4f  Prec@1 %.4f  Prec@5 %.4f'
                   %(epoch + 1, i + 1, len(trainloader), elapsed,
                    losses.val, top1.val, top5.val))
            sys.stdout.write("\b" * (len(prt_str))) 
            sys.stdout.write(prt_str)
            sys.stdout.flush()

        print_log(
            '\n    **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                  error1=100 - top1.avg), log)
            # if i % 100 == 99:    
            #     # print statistics every 100 mini-batches each epoch
            #     progress.display(i) # i = batch id in the epoch

        # update the learning rate
        scheduler.step()

        # print test accuracy every few epochs
        if epoch % 1 == 0:
            # print_log('epoch {}'.format(epoch+1), log)
            epoch_acc, epoch_los = test_accu(testloader, net, device, log)
            if 'pg' in _ARCH:
                sparsity(testloader, net, device)
            if epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(net.state_dict())
            # print_log("The best test accuracy so far: {:.1f}".format(best_acc), log)

            # save the model if required
            if args.save:
                print_log("\r    Saving the trained model and states.", log)
                # this_file_path = os.path.dirname(os.path.abspath(__file__))
                # save_folder = os.path.join(this_file_path, 'save_CIFAR10_model')
                util.save_models(best_model, save_folder,
                        suffix=dt_str+_ARCH+'-finetune' if args.finetune else dt_str+_ARCH)
                """
                states = {'epoch':epoch+1, 
                          'optimizer':optimizer.state_dict(), 
                          'scheduler':scheduler.state_dict()}
                util.save_states(states, save_folder, suffix=_ARCH)
                """
        train_los = losses.avg
        train_acc = top1.avg
        val_los = epoch_los
        val_acc = epoch_acc
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)

        epoch_time.update(time.time() - start_time)
        start_time = time.time()

        recorder.plot_curve(os.path.join(save_folder, 'curve.png'))

    print_log('Finished Training', log)

#----------------------------
# Test accuracy.
#----------------------------
def test_accu(testloader, net, device, log, num_iter = 1):
    net.to(device)
    correct = 0
    total = 0
    running_loss = 0.0
    # switch the model to the evaluation mode
    net.eval()
    with torch.no_grad():
        iter_count = 0
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            criterion = (nn.CrossEntropyLoss().cuda() 
                if torch.cuda.is_available() else nn.CrossEntropyLoss())
            loss = criterion(outputs, labels)
            if 'pg' in _ARCH:
                for name, param in net.named_parameters():
                    if 'threshold' in name:
                        loss += 0.00001 * 0.5 * torch.norm(param-args.gtarget) * torch.norm(param-args.gtarget)
            running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            iter_count += 1
            # if iter_count == num_iter:
            #     break

    accuracy = 100.0 * correct / total
    print_log('\r    Accuracy of the network on the test images: %.1f %%' % (accuracy), log)
    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    return val_acc, val_loss


dataset = radioml_18_dataset(args.data_dir)

#----------------------------
# Log
#----------------------------
def print_log(print_string, log):
    text = '{}'.format(print_string)
    print(text)
    log.write('{}\n'.format(print_string))
    log.flush()


#----------------------------
# Main function.
#----------------------------

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.which_gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    log = open(os.path.join(args.save_folder, 'log.txt'), 'w')
    print_log('save path : {}'.format(args.save_folder), log)
    # state = {k: v for k, v in args}
    # print_log(state, log)
    print_log("python version : {}".format(
        sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(
        torch.backends.cudnn.version()), log)

    print_log("Available GPUs: {}".format(torch.cuda.device_count()), log)

    print_log("Create model.", log)

    # net = models.resnet18(pretrained=True)
    # net.fc = nn.Linear(512, args.num_classes)
    net = UltraNet()
    # net = generate_model(_ARCH)
    print_log("{} \n".format(net), log)
    n_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print_log("Number of parameters: {}\n".format(n_param), log)


    if torch.cuda.device_count() > 1:
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        print_log("Activate multi GPU support.", log)
        net = nn.DataParallel(net)
    net.to(device)

    #------------------
    # Load model params
    #------------------
    if args.resume is not None or args.test:
        model_path = args.resume
        if os.path.exists(model_path):
            print_log("Loading trained model from {}.".format(model_path), log)
            net.load_state_dict(torch.load(model_path), strict=True)
        else:
            raise ValueError("Model not found.")

    #-----------------
    # Prepare Data
    #-----------------
    print_log("Loading data.", log)
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.train_sampler)
    testloader = DataLoader(dataset, batch_size=batch_size, sampler=dataset.test_sampler)

    #-----------------
    # Test
    #-----------------
    if args.test:
        print_log("Mode: Test only.", log)
        test_accu(testloader, net, device, log)

    #-----------------
    # Finetune
    #-----------------
    elif args.finetune:
        print_log("num epochs = {}".format(num_epoch), log)
        # initial_lr = 1e-5
        print_log("init lr = {}".format(args.learning_rate), log)
        optimizer = optim.Adam(net.parameters(),
                          lr = args.learning_rate,
                          weight_decay=0.)
        lr_decay_milestones = [100, 150, 200] #[150, 250, 300]#[200, 250, 300]#[50, 100] 
        print_log("milestones = {}".format(lr_decay_milestones), log)
        scheduler = optim.lr_scheduler.MultiStepLR(
                            optimizer,
                            milestones=lr_decay_milestones,
                            gamma=0.1,
                            last_epoch=_LAST_EPOCH)
        start_epoch=0
        print_log("Start finetuning.", log)
        train_model(trainloader, testloader, net,
                    optimizer, scheduler, start_epoch, device, log)
        test_accu(testloader, net, device, log, num_iter = 1)

    #-----------------
    # Train
    #-----------------
    else:
        #-----------
        # Optimizer
        #-----------
        # initial_lr = 1e-3
        optimizer = optim.Adam(net.parameters(),
                          lr = args.learning_rate,
                          weight_decay=args.decay)

        #-----------
        # Scheduler
        #-----------
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
        
        start_epoch = 0
        print_log("Start training.", log)
        train_model(trainloader, testloader, net, 
                    optimizer, scheduler, start_epoch, device, log)
        test_accu(testloader, net, device, log)

if __name__ == "__main__":
    main()
