from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from davis_2017 import TripletImageLoader
import siamese_net as models
from tripletnet import Tripletnet
import matplotlib.pyplot as plt
from skimage import io
from visdom import Visdom
import numpy as np
import random
from utils import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--lr-decay', type=float, default=0.5, metavar='LD',
                    help='learning rate decay (default: 0.5)')
parser.add_argument('--decay-epoch', type=int, default=2, metavar='DS',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')
parser.add_argument('--arch', default='vgg16', type=str,
                    help='siamese net arch')
parser.add_argument('--pretrained', default=True, type=bool,
                    help='siamese is pretrained or not')

best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Normalize on RGB Value
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.arch.startswith('inception'):
        size = (299, 299)
    else:
        size = (224, 256)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    transform=transforms.Compose([
                            transforms.Resize((size[0],size[0])),
                            transforms.ToTensor(),
                            normalize,
                        ])

    # test_loader = torch.utils.data.DataLoader(
    #     TripletImageLoader('../video_segmentation/multi_mask', train=False, 
    #                     transform=transforms.Compose([
    #                         transforms.Resize((size[0],size[0])),
    #                         transforms.ToTensor(),
    #                         normalize,
    #                     ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)

    print("=> creating model '{}'".format(args.arch))
    model = models.setup(args)

    tnet = Tripletnet(model, args)
    print('load checkpoint...')
    tnet.load_state_dict(torch.load('./out/model_best.pth.tar')['state_dict'])
    # tnet.load_state_dict(torch.load('./out/TripletNet/checkpoint.pth.tar')['state_dict'])
    print('load checkpoint finish!')

    if args.cuda:
        tnet.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    # optimizer = optim.SGD(tnet.parameters(), lr=args.lr, momentum=args.momentum)
    # shaduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)

    #single img test 
    davis_path = '../video_segmentation/DAVIS2017/trainval'
    seqname = 'dancing'
    img_num = len(os.listdir(os.path.join(davis_path, 'JPEGImages', '480p', seqname)))

    img_id1 = random.randint(0,img_num-1)
    img_id2 = random.randint(0,img_num-1)

    obj1 = 1
    obj2 = 3

    img1 = io.imread(os.path.join(davis_path, 'JPEGImages', '480p', seqname, str(img_id1).zfill(5)+'.jpg'))
    mask1,_ = load_mask(os.path.join(davis_path, 'Annotations', '480p', seqname, str(img_id1).zfill(5)+'.png'), obj1)
    bbox1 = compute_bbox_from_mask(mask1)
    img1 = img1[bbox1[1]:bbox1[3],bbox1[0]:bbox1[2],:]

    img2 = io.imread(os.path.join(davis_path, 'JPEGImages', '480p', seqname, str(img_id2).zfill(5)+'.jpg'))
    mask2,_ = load_mask(os.path.join(davis_path, 'Annotations', '480p', seqname, str(img_id2).zfill(5)+'.png'), obj1)
    bbox2 = compute_bbox_from_mask(mask2)
    img2 = img2[bbox2[1]:bbox2[3],bbox2[0]:bbox2[2],:]

    img3 = io.imread(os.path.join(davis_path, 'JPEGImages', '480p', seqname, str(img_id1).zfill(5)+'.jpg'))
    mask3,_ = load_mask(os.path.join(davis_path, 'Annotations', '480p', seqname, str(img_id1).zfill(5)+'.png'), obj2)
    bbox3 = compute_bbox_from_mask(mask3)
    img3 = img3[bbox3[1]:bbox3[3],bbox3[0]:bbox3[2],:]

    plt.figure()
    plt.title('display')
    plt.subplot(131)
    plt.imshow(img1)
    plt.subplot(132)
    plt.imshow(img2)
    plt.subplot(133)
    plt.imshow(img3)
    plt.show()

    img1 = Image.fromarray(img1, 'RGB')
    img2 = Image.fromarray(img2, 'RGB')
    img3 = Image.fromarray(img3, 'RGB')

    img1 = transform(img1).unsqueeze(0)
    img2 = transform(img2).unsqueeze(0)
    img3 = transform(img3).unsqueeze(0)

    img1, img2, img3 = Variable(img1.cuda()), Variable(img2.cuda()), Variable(img3.cuda())

    tnet.eval()
    dista, distb, _, _, _ = tnet(img1, img3, img2)
    print('far distance: ', dista)
    print('close distance: ', distb)



    # n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    # print('  + Number of params: {}'.format(n_parameters))
    # i=0

    # # checkpoint_epoch = 13
    # # print('load checkpoint '+str(checkpoint_epoch))
    # # tnet.load_state_dict(torch.load('./out/TripletNet/checkpoint.pth.tar')['state_dict'])
    # # best_acc = torch.load('./out/model_best.pth.tar')['best_prec1']
    

    # for epoch in range(1, args.epochs + 1):
    #     if (i) % args.decay_epoch == 0:
    #         shaduler.step()

    #     # if epoch <= checkpoint_epoch:
    #     #     continue

    #     # train for one epoch
    #     train(train_loader, tnet, criterion, optimizer, epoch)
    #     # evaluate on validation set
    #     acc = test(test_loader, tnet, criterion, epoch)

    #     # remember best acc and save checkpoint
    #     is_best = acc > best_acc
    #     best_acc = max(acc, best_acc)
    #     save_checkpoint({
    #         'epoch': epoch + 1,
    #         'state_dict': tnet.state_dict(),
    #         'best_prec1': best_acc,
    #     }, is_best)
    #     i+=1

def train(train_loader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # switch to train mode
    tnet.train()
    for batch_idx, (data1, data2, data3) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)
        # print(batch_idx, len(train_loader.dataset))

        # compute output
        dista, distb, embedded_x, embedded_y, embedded_z = tnet(data1, data2, data3)
        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        
        loss_triplet = criterion(dista, distb, target)
        loss_embedd = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
        loss = loss_triplet + 0.001 * loss_embedd

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss_triplet.data[0], data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0]/3, data1.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f}) \t'
                  'lr: {:.8f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg,
                optimizer.state_dict()['param_groups'][0]['lr']))
    # log avg values to somewhere
    # plotter.plot('acc', 'train', epoch, accs.avg)
    # plotter.plot('loss', 'train', epoch, losses.avg)
    # plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)

def test(test_loader, tnet, criterion, epoch):
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to evaluation mode
    tnet.eval()
    for batch_idx, (data1, data2, data3) in enumerate(test_loader):
        if args.cuda:
            data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
        data1, data2, data3 = Variable(data1), Variable(data2), Variable(data3)

        # compute output
        dista, distb, _, _, _ = tnet(data1, data2, data3)
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        test_loss =  criterion(dista, distb, target).data[0]

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        accs.update(acc, data1.size(0))
        losses.update(test_loss, data1.size(0))      

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        losses.avg, 100. * accs.avg))
    # plotter.plot('acc', 'test', epoch, accs.avg)
    # plotter.plot('loss', 'test', epoch, losses.avg)
    return accs.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "out/%s/"%(args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'out/'+'model_best.pth.tar')

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum()*1.0/dista.size()[0]

if __name__ == '__main__':
    main()    
