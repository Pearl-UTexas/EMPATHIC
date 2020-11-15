import sys
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.distributions.normal import Normal

from data_loader_all import FaceFeatureDataset
from network_facs import Net

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Test the Model
def test_model(epoch, net, num_classes1, dataset, data_loader, regression, batch_size, validation=False, fix_dist=None):
    set_name = 'test'
    if validation: set_name = 'validation'
    net.eval()
    total = 0
    total_loss1 = 0.0
    total_loss2 = 0.0
    total_loss3 = 0.0

    for (img_features, labels, _, event_mask, binary_labels, annot_target) in data_loader:
        img_features = Variable(img_features.float()).cuda()
        event_mask = Variable(event_mask.float()).cuda()
        labels = Variable(labels).cuda()
        binary_labels = Variable(binary_labels).cuda()
        annot_target = Variable(annot_target).cuda()

        # Regression part is used for predicting other RL statistics such as Q-values and advantage.
        # For this paper, we only showed an instantiation using reward as target, so you should not use regression when running our code.
        if regression:           
            outputs_mu, outputs_sigma, outputs_rec = net(img_features.float(), event_mask)
            dist = Normal(loc=outputs_mu, scale=outputs_sigma)
            loss1 = 0.005*torch.mean(-dist.log_prob(labels.float()))
            loss2 = mseLoss(outputs_mu.squeeze(), labels.float())
        else:
            # loss1: cross entropy loss for 3 reward classes
            # loss2: binary cross entropy loss when lumping -5/-1 rewards into a single class of negative reward
            # loss3: MSE loss in auxiliary task for predicting the annotation labels. 

            if not fix_dist is None:
                outputs = np.tile(np.log(fix_dist), labels.shape[0]).reshape(labels.shape[0], -1)
                outputs = torch.Tensor(outputs).cuda().float()
            else:
                outputs, annot_pred = net(img_features.float(), event_mask)                      
                loss3 = mseLoss(annot_pred.squeeze(), annot_target.float())

            binary_outputs = torch.stack((torch.log(torch.sum(torch.exp(outputs[:,0:2]), dim=1)), outputs[:,2]), 1) 
            loss1 = ceLoss_test(outputs, labels.long())
            loss2 = loss1 # ceLoss_test(binary_outputs, binary_labels.long())
            _, predicted = torch.max(outputs.data, 1)
 
        data_ct = labels.size(0)
        total += data_ct
        total_loss1 += loss1.data.item()*data_ct
        total_loss2 += loss2.data.item()*data_ct
        if fix_dist is None: total_loss3 += loss3.data.item()*data_ct

    if total == 0: 
        print('WARN! No data in '+set_name+'!')
        return 0
    avg_loss1 = float(total_loss1)/float(total)
    avg_loss2 = float(total_loss2)/float(total)
    avg_loss3 = float(total_loss3)/float(total)

    if regression:
        if validation:
            print('Log PDF Loss on validation set: %f' % avg_loss1)
            print('MSE on validation set: %f' % avg_loss2)
        else:
            print('Log PDF Loss on test set: %f' % avg_loss1)
            print('MSE on test set: %f' % avg_loss2)
    else:
        model = 'network'
        if not fix_dist is None: model = 'fix_distribution'
        print('Cross Entropy Loss (Reward) of ' + model + ' on ' + set_name + ' set: %f' % avg_loss1)
        print('Cross Entropy Loss (Reward) of binary ' + model + ' on ' + set_name + ' set: %f' % avg_loss2)
        print('MSE Loss (Reward) of ' + model + ' on ' + set_name + ' set: %f' % avg_loss3)

    if regression:
        return avg_loss2
    else:
        return avg_loss1

# Hyper Parameters 
num_epochs = 100
batch_size = 8

frames_pre=4
frames_after=18
window_size = frames_pre + frames_after + 1

apply_event_mask = False
include_binary_loss = True
include_aux_loss = True
threshold_percentile = 0

# index of the target task statistic for predicting.
# Each line in data file starts with timestep index of the episode.
# Following: 1: reward; 2: optimality; 3: qval_opt; 4: qval_beh; 5: adv_opt; 6: adv_beh 7: suprise
target_idx = 1
if target_idx < 3: regression = False
else: regression = True
learning_rate = 0.002

# Just a placeholder when training on all subjects for a single model.
subject_under_test = "WkOsToXr9v" 
num_classes1 = 3
num_classes2 = 10 * window_size

# Loss and Optimizer
ceLoss_train = nn.CrossEntropyLoss()
ceLoss_test = nn.CrossEntropyLoss()
mseLoss = nn.MSELoss()

train_dataset = FaceFeatureDataset(subject_under_test=subject_under_test, target_idx=target_idx,  data_dir='detected/', threshold_percentile=threshold_percentile, regression=regression, frames_pre=frames_pre, frames_after=frames_after)

test_dataset = FaceFeatureDataset(subject_under_test=subject_under_test, target_idx=target_idx , test=True,  data_dir='detected/', threshold_percentile=threshold_percentile, balance_data=False, regression=regression, frames_pre=frames_pre, frames_after=frames_after)

# Data Loader (Input Pipeline)
if regression: 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, drop_last=True)
else: 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, sampler=train_dataset.sampler, drop_last=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size)

# Initialize Neural net
net = Net(input_size=train_dataset.feature_size, num_classes1=num_classes1, num_classes2=num_classes2, apply_event_mask=apply_event_mask, regression=regression, window_size=window_size)
net.cuda()

print(net)

if learning_rate == 0: optimizer = torch.optim.Adadelta(net.parameters())  
else: optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
lowest_loss = float('inf')
best_model_wts = None
        
# Get baseline loss performance of fixed distribution
if not regression:
    label_distribution = train_dataset.label_distribution
    print('Label Distribution:',label_distribution)
    baseline_loss_test = test_model(0, net, num_classes1, test_dataset, test_loader, regression, batch_size, fix_dist=label_distribution)

# Train the Model
for epoch in range(num_epochs):
    if epoch % 2 == 0:
        test_loss = test_model(epoch, net, num_classes1, test_dataset, test_loader, regression, batch_size)
        # Save the Model at specified dir
        if test_loss < lowest_loss:
            lowest_loss = test_loss
            best_model_wts = copy.deepcopy(net.state_dict())           
    if epoch == num_epochs - 1: break
    if epoch > 0 and epoch % 25 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8

    net.train()
    for i, (img_features, labels, _, event_mask, binary_labels, annot_targets) in enumerate(train_loader):  
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        img_features = Variable(img_features.float()).cuda()
        event_mask = Variable(event_mask.float()).cuda()
        labels = Variable(labels).cuda()
        binary_labels = Variable(binary_labels).cuda()
        annot_targets = Variable(annot_targets).cuda()

        if regression:           
            outputs_mu, outputs_sigma, outputs_rec = net(img_features.float(), event_mask)
            dist = Normal(loc=outputs_mu, scale=outputs_sigma)
            loss1 = 0.005*torch.mean(-dist.log_prob(labels.float()))
            loss2 = mseLoss(outputs_mu.squeeze(), labels.float())
            loss = loss1 + loss2
        else:
            outputs, outputs_rec = net(img_features.float(), event_mask)           
            binary_outputs = torch.stack((torch.log(torch.sum(torch.exp(outputs[:,0:2]), dim=1)), outputs[:,2]), 1) 
            
            loss1 = ceLoss_train(outputs, labels.long())
            loss2 = ceLoss_train(binary_outputs, binary_labels.long())       
            loss3 = mseLoss(outputs_rec, annot_targets.float())
            if include_aux_loss: loss = loss1 + 2*loss2 + loss3
            else: loss = loss1 + 2*loss2
        
        loss.backward()
        optimizer.step()
        if (i+1) % 20 == 0:
            if regression:
                 print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, LogPDF_Loss: %.4f, Auxiliary_Loss: %.4f' 
               %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item(), loss1.data.item(), loss2.data.item()))
            else:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Classification_Loss: %.4f, Binary_Classification_Loss: %.4f, Auxiliary_loss: %.4f' 
               %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item(), loss1.data.item(),  loss2.data.item(), loss3.data.item()))

torch.save(best_model_wts, 'MLP_facs_reward_models/allsubjects_'+str(np.round(lowest_loss,4))+'.pkl')
