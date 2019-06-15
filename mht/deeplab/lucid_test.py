import socket
import timeit
from datetime import datetime
import os
import numpy as np
import glob
from collections import OrderedDict
from PIL import Image

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
from dataloaders import lucid_dataset as db
from dataloaders import utils
from networks import deeplab_xception, deeplab_resnet
from net.generateNet import generate_net
from dataloaders import custom_transforms as tr
from dataloaders.utils import *

torch.cuda.manual_seed(1701)

gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
use_sbd = False  # Whether to use SBD dataset
nEpochs = 110  # Number of epochs for training
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
seq_name_list = [#'carousel', 'cats-car', 'chamaleon', 'deer', 'giant-slalom',
                 #'girl-dog', 'golf', 'guitar-violin', 
                 'gym', 'helicopter', 'horsejump-stick', 'hoverboard',
                 'lock', 'man-bike', 'monkeys-trees', 'mtb-race', 'orchid', 'people-sunset', 'planes-crossing',
                 'rollercoaster', 'salsa', 'seasnake', 'skate-jump', 'slackline', 
                 'subway', 'tandem','tennis-vest', 'tractor']
# obj_id = 1

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

# Network definition
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

# net.load_state_dict(torch.load('./run/mask_epoch-4.pth'))
# net.load_state_dict(torch.load('./two.pth'))

modelName = 'deeplabv3plus-' + backbone + '-voc'
criterion = nn.BCEWithLogitsLoss()
# criterion = utils.class_balanced_cross_entropy_loss

def im_normalize(im):
    """
    Normalize image
    """
    imn = (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn

def get_obj_num(path):
    label = Image.open(path)
    mask = np.atleast_3d(label)[...,0]
    obj_num = len(np.unique(mask))
    return obj_num


if resume_epoch == 0:
    print("Training deeplabv3+ from scratch...")
else:
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

if resume_epoch != nEpochs:
    # Logging into Tensorboard
    # log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    # writer = SummaryWriter(log_dir=log_dir)

    for seq_name in seq_name_list:
        obj_num = get_obj_num('../DAVIS2017/test_dev/Annotations/480p/'+seq_name+'/00000.png')

        for obj_id in range(obj_num):
            if obj_id == 0:
                continue

            if seq_name == 'guitar-violin' and obj_id==1:
                continue

            print('train for obj '+str(obj_id)+' of '+seq_name)

            # Use the following optimizer
            # net.load_state_dict(torch.load('./run/mask_34.pth'))
            net.load_state_dict(torch.load('./run/mask_final.pth'))
            # pretrained_dict = torch.load('./model/carousel_1.pth')
            # pretrained_dict_copy = pretrained_dict.copy()
            # # print(pretrained_dict['module.backbone.conv1.weight'])
            # # print(pretrained_dict['module.aspp.branch3.1.running_mean'])
            # # print(pretrained_dict.keys())
            # # pretrained_dict.pop("conv11.weight")
            # net_dict = net.state_dict()
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape==net_dict[k].shape)}
            # net_dict.update(pretrained_dict)
            # # net_dict['module.backbone.conv1.weight'][:,0:3,:,:] = pretrained_dict_copy['module.backbone.conv1.weight']
            # # print(net_dict['module.backbone.conv1.weight'])
            # # print(net_dict['module.aspp.branch3.1.running_mean']==pretrained_dict['module.aspp.branch3.1.running_mean'])
            # net.load_state_dict(net_dict)
            # net.load_state_dict(torch.load('./model/carousel_1.pth'))
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
            db_train = db.OnlineDataset(train=True, transform=composed_transforms, seq_name = seq_name, obj_id=obj_id)
            trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0) 

            # Testing dataset and its iterator
            # db_test = db.OnlineDataset(train=False, transform=tr.ToTensor(), seq_name = seq_name)
            # testloader = DataLoader(db_test, batch_size=testBatch, shuffle=False, num_workers=0)  

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
            # num_img_ts = len(testloader)
            running_loss_tr = 0.0
            running_loss_ts = 0.0
            aveGrad = 0
            global_step = 0
            print("Training Network")   

      
            # Main Training and Testing Loop
            for epoch in range(resume_epoch, nEpochs):
                start_time = timeit.default_timer() 

                if epoch % p['epoch_size'] == p['epoch_size'] - 1:
                # if epoch%1==0:
                    lr_ = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
                    print('(poly lr policy) learning rate: ', lr_)
                    optimizer = optim.Adam(net.parameters(), lr=lr_, weight_decay=p['wd']) 

                if epoch <= 60:
                    continue  

                #net.train()
                #for ii, sample_batched in enumerate(trainloader):
                #    print('Training || epoch: ', epoch, ' || iter: ', ii)
                #    inputs, labels = sample_batched['image'], sample_batched['label']
                #    # print(inputs.shape)
                #    # Forward-Backward of the mini-batch
                #    inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                #    global_step += inputs.data.shape[0] 

                #    if gpu_id >= 0:
                #        inputs, labels = inputs.cuda(), labels.cuda()   

                #    outputs = net.forward(inputs)   

                #    loss = criterion(outputs, labels)
                #    running_loss_tr += loss.item()  

                #    # Print stuff
                #    if ii % num_img_tr == (num_img_tr - 1):
                #        running_loss_tr = running_loss_tr# / num_img_tr
                #        # writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                #        print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                #        print('Loss: %f' % running_loss_tr)
                #        running_loss_tr = 0
                #        stop_time = timeit.default_timer()
                #        print("Execution time: " + str(stop_time - start_time) + "\n")    

                #    # Backward the averaged gradient
                #    loss /= p['nAveGrad']
                #    loss.backward()
                #    aveGrad += 1    

                #    # Update the weights once in p['nAveGrad'] forward passes
                #    if aveGrad % p['nAveGrad'] == 0:
                #        # writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                #        optimizer.step()
                #        optimizer.zero_grad()
                #        aveGrad = 0

                if epoch%10 == 9:
                    net.load_state_dict(torch.load('./model/'+seq_name+'_'+str(obj_id)+'_'+str(epoch)+'.pth'))
                    print("load  model at {}\n".format('./model/'+seq_name+'_'+str(obj_id)+'_'+str(epoch)+'.pth'))  

                #     # Show 10 * 3 images results each epoch
                #     if ii % (num_img_tr // 10) == 0:
                #         grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                #         writer.add_image('Image', grid_image, global_step)
                #         grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
                #                                range=(0, 255))
                    #         writer.add_image('Predicted label', grid_image, global_step)
                #         grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
                #         writer.add_image('Groundtruth label', grid_image, global_step)    

                # # Save the model
                # if (epoch % snapshot) == snapshot - 1:
                #     torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
                #     print("Save model at {}\n".format(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))) 

                # One testing epoch
                # if useTest and epoch % nTestInterval == (nTestInterval - 1):

                    total_miou = 0.0        

                    import matplotlib.pyplot as plt
                    plt.close("all")
                    plt.ion()
                    f, ax_arr = plt.subplots(1, 3)  

                    img_path = np.sort(os.listdir(os.path.join('../DAVIS2017/test_dev/JPEGImages/480p', seq_name)))
                    img_list = list(map(lambda x: os.path.join('../DAVIS2017/test_dev/JPEGImages/480p', seq_name, x), img_path))[1:]        

                    mask_path = os.path.join('../DAVIS2017/test_dev/Annotations/480p', seq_name)
                    mask = os.path.join(mask_path, '00000.png')     

                    label_path = np.sort(os.listdir(os.path.join('../DAVIS2017/test_dev/Annotations/480p', seq_name)))
                    label_list = os.path.join('../DAVIS2017/test_dev/Annotations/480p', seq_name, label_path[0])
                    # label_list = list(map(lambda x: os.path.join('../DAVIS2017/test_dev/Annotations/480p', seq_name, x), label_path))[1:]          

                    f_mask_ ,_ = utils.load_mask(mask, obj_id)
                    # _, palette = utils.imread_indexed(mask)
                    # print(palette.shape)
                    f_mask = f_mask_.copy()
                    mask = np.array(f_mask, dtype=np.float32)  
                    h, w = mask.shape     

                    # flow_path = np.sort(os.listdir(os.path.join('../all_test_file/test_flow', seq_name)))
                    # flow_list = list(map(lambda x: os.path.join('../all_test_file/test_flow', seq_name, x), flow_path))       

                    net.eval()
                    for ii in range(len(img_list)):
                        if mask.max() == 0:
                            break
                        # print('Testing || epoch: ', epoch, ' || iter: ', ii)
                        # mask = mask*255
                        mask = np.array(mask, dtype=np.float32)

                        im2_id = int(img_list[ii].split('/')[-1].split('.')[0])
                        im1_id = im2_id - 1
                        obj_id = obj_id

                        flow_dir = os.path.join('../all_test_file/test_flow', seq_name)
                        img_dir = os.path.join('../DAVIS2017/test_dev/JPEGImages/480p', seq_name)
                        warped_mask, validflowmap01,_,_ = warp_mask(mask, im1_id, im2_id, flow_dir, img_dir)
                        warped_mask = (warped_mask > 0.3).astype(np.float32)
                        if warped_mask.max()==0:
                            break

                        inputs, labels, bbox, crop_size, palette = utils.make_img_gt_pair(img_list[ii], mask, seq_name, label_list, obj_id)
                        # inputs, labels = tr.ToTensor()(inputs), tr.ToTensor()(labels)
                        # Forward pass of the mini-batch
                        with torch.no_grad():
                            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
                            if gpu_id >= 0:
                                inputs, labels = inputs.cuda(), labels.cuda()       

                        
                            outputs = net.forward(inputs)       

                            for jj in range(int(inputs.size()[0])):
                                pred = np.transpose(outputs.cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
                                pred = 1 / (1 + np.exp(-pred))
                                pred = np.squeeze(pred)
                                mask_local = utils.valid_mask(pred)
                                mask_local = cv2.resize(mask_local, crop_size, interpolation=cv2.INTER_NEAREST)
                                mask = np.zeros((h,w)).astype(np.float32)
                                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask_local
                        # print(mask.shape)     

                        # img_ = np.transpose(inputs.cpu().numpy()[jj, :, :, :], (1, 2, 0))
                        # # print(img_.shape)
                        # gt_ = np.transpose(labels.cpu().numpy()[jj, :, :, :], (1, 2, 0))
                        # gt_ = np.squeeze(labels.cpu())
                        # # Plot the particular example
                        # ax_arr[0].cla()
                        # ax_arr[1].cla()
                        # ax_arr[2].cla()
                        # ax_arr[0].set_title('Input Image')
                        # ax_arr[1].set_title('Ground Truth')
                        # ax_arr[2].set_title('Detection')
                        # ax_arr[0].imshow(im_normalize(img_[:,:,:3]))
                        # ax_arr[1].imshow(gt_)
                        # ax_arr[2].imshow(im_normalize(pred))
                        # plt.pause(0.01)     

                            if not os.path.exists('../all_test_file/mask_result'):
                                os.mkdir('../all_test_file/mask_result')
                            if not os.path.exists('../all_test_file/mask_result/'+seq_name):
                                os.mkdir('../all_test_file//mask_result/'+seq_name)
                            if not os.path.exists('../all_test_file/mask_result/'+seq_name+'/'+str(obj_id)):
                                os.mkdir('../all_test_file/mask_result/'+seq_name+'/'+str(obj_id))
                            if not os.path.exists('../all_test_file/mask_result/'+seq_name+'/'+str(obj_id)+'/'+str(epoch)):
                                os.mkdir('../all_test_file/mask_result/'+seq_name+'/'+str(obj_id)+'/'+str(epoch))
                            save_path = os.path.join('../all_test_file/mask_result',seq_name,str(obj_id),str(epoch),str(ii+1).zfill(5)+'.png')
                    # print(mask)
                            utils.imwrite_index(save_path, mask, palette)       

                        # loss = criterion(outputs, labels, size_average=False)
                        # running_loss_ts += loss.item()
            #     predictions = torch.max(outputs, 1)[1]
            #     total_miou += utils.get_iou(predictions, labels)
            # miou = total_miou / (ii * testBatch + inputs.data.shape[0])
            # print('miou: ', miou) 
            print('train for obj '+str(obj_id)+' of '+seq_name+' finish!')   
        # break   

                        # # Print stuff
                        # if ii % num_img_ts == num_img_ts - 1:     

                        #     miou = total_miou / (ii * testBatch + inputs.data.shape[0])
                        #     running_loss_ts = running_loss_ts / num_img_ts        

                        #     print('Validation:')
                        #     print('[Epoch: %d, numImages: %5d]' % (epoch, ii * testBatch + inputs.data.shape[0]))
                        #     writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
                        #     writer.add_scalar('data/test_miour', miou, epoch)
                        #     print('Loss: %f' % running_loss_ts)
                        #     print('MIoU: %f\n' % miou)
                        #     running_loss_ts = 0   
            # writer.close()
