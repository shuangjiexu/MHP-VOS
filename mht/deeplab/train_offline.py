import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict

# PyTorch includes
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import pascal, sbd, combine_dbs
from dataloaders import davis_2017 as db
from dataloaders import utils
from networks import deeplab_xception, deeplab_resnet
from net.generateNet import generate_net
from dataloaders import custom_transforms as tr
from matplotlib import pyplot as plt

torch.cuda.manual_seed(1701)

gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
use_sbd = False  # Whether to use SBD dataset
nEpochs = 100  # Number of epochs for training
resume_epoch = 0   # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
p['trainBatch'] = 2  # Training batch size
testBatch = 1  # Testing batch size
useTest = True  # See evolution of the test set when training
nTestInterval = 5 # Run on test set every nTestInterval epochs
snapshot = 10  # Store a model every snapshot epochs
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-5  # Learning rate
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.9  # Momentum
p['epoch_size'] = 10  # How many epochs to change learning rate
backbone = 'resnet' # Use xception or resnet as feature extractor,

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run')

# # Network definition
# if backbone == 'xception':
#     net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=1, os=16, pretrained=True)
# elif backbone == 'resnet':
#     net = deeplab_resnet.DeepLabv3_plus(nInputChannels=6, n_classes=1, os=16, pretrained=True)
# else:
#     raise NotImplementedError
net = generate_net()
device = torch.device('cuda')
net = nn.DataParallel(net)
net.to(device)

modelName = 'deeplabv3plus-' + backbone + '-voc'
# criterion = utils.class_balanced_cross_entropy_loss
criterion = nn.BCEWithLogitsLoss()


if resume_epoch == 0:
    print("Training deeplabv3+ from scratch...")
else:
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU

print('loading weights from pretrained coco/voc datasets')
pretrained_dict = torch.load('./deeplabv3plus_res101_atrous_VOC2012_epoch60_all.pth')
pretrained_dict_copy = pretrained_dict.copy()
# print(pretrained_dict['module.backbone.conv1.weight'])
# print(pretrained_dict['module.aspp.branch3.1.running_mean'])
# print(pretrained_dict.keys())
# pretrained_dict.pop("conv11.weight")
net_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
net_dict.update(pretrained_dict)
net_dict['module.backbone.conv1.weight'][:,0:3,:,:] = pretrained_dict_copy['module.backbone.conv1.weight']
# print(net_dict['module.backbone.conv1.weight'])
# print(net_dict['module.aspp.branch3.1.running_mean']==pretrained_dict['module.aspp.branch3.1.running_mean'])
net.load_state_dict(net_dict)
print('loading weights finish!')

# if gpu_id >= 0:
#     torch.cuda.set_device(device=gpu_id)
#     net.cuda()

if resume_epoch != nEpochs:
    # Logging into Tensorboard
    # log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    # writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    # optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    optimizer = optim.Adam(net.parameters(), lr=p['lr'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    # composed_transforms_tr = transforms.Compose([
    #     tr.RandomSized(512),
    #     tr.RandomRotate(15),
    #     tr.RandomHorizontalFlip(),
    #     tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     tr.ToTensor()])

    # composed_transforms_ts = transforms.Compose([
    #     tr.FixedResize(size=(512, 512)),
    #     tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #     tr.ToTensor()])

    # voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr)
    # voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)
    # Define augmentation transformations as a composition
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tr.ToTensor()])
    # Training dataset and its iterator
    db_train = db.OfflineDataset(train=True, transform=composed_transforms)
    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0)

    # Testing dataset and its iterator
    db_test = db.OfflineDataset(train=False, transform=transforms.Compose([tr.ToTensor()]))
    testloader = DataLoader(db_test, batch_size=testBatch, shuffle=False, num_workers=0)

    if use_sbd:
        print("Using SBD dataset")
        sbd_train = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr)
        db_train = combine_dbs.CombineDBs([voc_train, sbd_train], excluded=[voc_val])
    else:
        db_train = db_train

    # trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0)
    # testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=0)

    # utils.generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    # print(num_img_tr)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    best_loss = 1000000000000.
    print("Training Network")

    checkpoint_epoch = 74
    print('load checkpoint '+str(checkpoint_epoch))
    net.load_state_dict(torch.load('./run/mask_74.pth'))

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        print('Training || epoch: ', epoch)
        start_time = timeit.default_timer()

        if epoch % p['epoch_size'] == p['epoch_size'] - 1:
            lr_ = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            optimizer = optim.Adam(net.parameters(), lr=lr_, weight_decay=p['wd'])

        if epoch <= checkpoint_epoch:
            continue

        net.train()
        # print(len(trainloader))
        for ii, sample_batched in enumerate(trainloader):
            # print('Training || epoch: ', epoch, ' || iter: ', ii)
            inputs, labels = sample_batched['image'], sample_batched['label']
            # inputs = inputs.squeeze(0)
            # labels = labels.squeeze(0)
            # print(inputs.shape)
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            global_step += inputs.data.shape[0]

            if gpu_id >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net.forward(inputs)
            # print(outputs.shape)

            loss = criterion(outputs, labels)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == (num_img_tr - 1):
                running_loss_tr = running_loss_tr# / num_img_tr
                # writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                # writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            # Show 10 * 3 images results each epoch
            # if ii % (num_img_tr // 10) == 0:
            #     grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
            #     writer.add_image('Image', grid_image, global_step)
            #     grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
            #                            range=(0, 255))
            #     writer.add_image('Predicted label', grid_image, global_step)
            #     grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
            #     writer.add_image('Groundtruth label', grid_image, global_step)

        # Save the model
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            torch.save(net.state_dict(), os.path.join(save_dir, 'mask' + '_' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'mask' + '_' + str(epoch) + '.pth')))
        # if (epoch % snapshot) == snapshot - 1:
        if epoch == nEpochs - 1:
            torch.save(net.state_dict(), os.path.join(save_dir, 'mask' + '_final' + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'mask' + '_final' + '.pth')))
        # One testing epoch
        # if useTest and epoch % nTestInterval == (nTestInterval - 1):
        #     total_miou = 0.0
        #     net.eval()
        #     print('Testing || epoch: ', epoch)
        #     for ii, sample_batched in enumerate(testloader):
        #         # print('Testing || epoch: ', epoch, ' || iter: ', ii)
        #         inputs, labels = sample_batched['image'], sample_batched['label']

        #         # Forward pass of the mini-batch
        #         with torch.no_grad():
        #             inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
        #             if gpu_id >= 0:
        #                 inputs, labels = inputs.cuda(), labels.cuda()

                
        #             outputs = net.forward(inputs)

        #         predictions = torch.max(outputs, 1)[1]

        #         loss = criterion(outputs, labels)
        #         running_loss_ts += loss.item()

        #         total_miou += utils.get_iou(predictions, labels)

        #         # Print stuff
        #         if ii % num_img_ts == num_img_ts - 1:
        #             # Save the model
        #             # if best_loss>running_loss_ts:
        #             #     best_loss = running_loss_ts
        #             torch.save(net.state_dict(), os.path.join(save_dir, 'mask' + '_' + str(epoch) + '.pth'))
        #             print("Save model at {}\n".format(os.path.join(save_dir, 'mask' + '_' + str(epoch) + '.pth')))

        #             miou = total_miou / (ii * testBatch + inputs.data.shape[0])
        #             running_loss_ts = running_loss_ts / num_img_ts

        #             print('Validation:')
        #             print('[Epoch: %d, numImages: %5d]' % (epoch, ii * testBatch + inputs.data.shape[0]))
        #             # writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
        #             # writer.add_scalar('data/test_miour', miou, epoch)
        #             print('Loss: %f' % running_loss_ts)
        #             print('MIoU: %f\n' % miou)
        #             running_loss_ts = 0

            
    # writer.close()
