import sys
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.distributions.normal import Normal

from data_loader_facs import FaceFeatureDataset
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

    for (img_features, labels, _, event_mask, binary_labels1, binary_labels2, annot_target) in data_loader:
        img_features = Variable(img_features.float()).cuda()
        event_mask = Variable(event_mask.float()).cuda()
        labels = Variable(labels).cuda()
        binary_labels1 = Variable(binary_labels1).cuda()
        binary_labels2 = Variable(binary_labels2).cuda()
        annot_target = Variable(annot_target).cuda()

        # Regression part is used for predicting other RL statistics such as Q-values and advantage.
        # For this paper, we only showed an instantiation using reward as target, so you should not use regression when running our code.
        if regression:           
            outputs_mu, outputs_sigma, annot_pred = net(img_features.float(), event_mask)
            dist = Normal(loc=outputs_mu, scale=outputs_sigma)
            loss1 = torch.mean(-dist.log_prob(labels.float()))
            loss2 = mseLoss(outputs_mu.squeeze(), labels.float())
            loss3 = mseLoss(annot_pred.squeeze(), annot_target.float())
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

            if num_classes1 == 3:
                binary_outputs1 = torch.stack((torch.log(torch.sum(torch.exp(outputs[:,0:2]), dim=1)), outputs[:,2]), 1) 
                binary_outputs2 = torch.stack((outputs[:,0], torch.log(torch.sum(torch.exp(outputs[:,1:3]), dim=1))), 1) 
            else:
                binary_outputs1 = outputs
                binary_outputs2 = outputs

            if binary_loss_option == 1:
                loss2 = ceLoss_test(binary_outputs1, binary_labels1.long())
            elif binary_loss_option == 2:
                loss2 = ceLoss_test(binary_outputs2, binary_labels2.long())
            elif binary_loss_option == 3:
                loss2 = 0.5 * (ceLoss_test(binary_outputs1, binary_labels1.long()) + ceLoss_test(binary_outputs2, binary_labels2.long()))

            loss1 = ceLoss_test(outputs, labels.long())
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
            print('Aux Loss on validation set: %f' % avg_loss3)
        else:
            print('Log PDF Loss on test set: %f' % avg_loss1)
            print('MSE on test set: %f' % avg_loss2)
            print('Aux Loss on test set: %f' % avg_loss3)
    else:
        model = 'network'
        if not fix_dist is None: model = 'fix_distribution'
        print('Cross Entropy Loss (Reward) of ' + model + ' on ' + set_name + ' set: %f' % avg_loss1)
        print('Cross Entropy Loss (Reward) of binary ' + model + ' on ' + set_name + ' set: %f' % avg_loss2)
        print('Aux Loss of ' + model + ' on ' + set_name + ' set: %f' % avg_loss3)

    if regression:
        return avg_loss2
    else:
        return avg_loss1

# Hyper Parameters 
num_epochs = 60
batch_size = 8
h_sizes=[128,128,64,8]
embed_size=64
dropout=0.63
learning_rate = 0.001

frames_pre=0
frames_after=12
log_PDF_loss_weight = 0.005
window_size = frames_pre + frames_after + 1

apply_event_mask = False
include_binary_loss = False
include_aux_loss = True 
binary_classification=False
data_fold_idx = 0
threshold_percentile = 0

# binary_loss_option: select the way of lumping the rewards into binary classification.
# 1: lumping -5/-1; 2: lumping -1/+6; 3: lumping both in 2 binary losses, with 0.5 weight each.
binary_loss_option = 1

# index of the target task statistic for predicting.
# Each line in data file starts with timestep index of the episode.
# Following: 1: reward; 2: optimality; 3: qval_opt; 4: qval_beh; 5: adv_opt; 6: adv_beh 7: suprise
target_idx = 1
if target_idx < 3: regression = False
else: regression = True

# The human subject ID you will train your model for.
subject_under_test = sys.argv[1] 
if binary_classification or target_idx == 2:
    num_classes1 = 2
else:
    num_classes1 = 3
num_classes2 = 10 * window_size

# Loss and Optimizer
ceLoss_train = nn.CrossEntropyLoss()
ceLoss_test = nn.CrossEntropyLoss()
mseLoss = nn.MSELoss()

train_dataset = FaceFeatureDataset(subject_under_test=subject_under_test, target_idx=target_idx,  data_dir='detected/', threshold_percentile=threshold_percentile, regression=regression, frames_pre=frames_pre, frames_after=frames_after, binary_classification=binary_classification, data_fold_idx=data_fold_idx)

test_dataset = FaceFeatureDataset(subject_under_test=subject_under_test, target_idx=target_idx , test=True,  data_dir='detected/', threshold_percentile=threshold_percentile, balance_data=False, regression=regression, frames_pre=frames_pre, frames_after=frames_after, binary_classification=binary_classification, data_fold_idx=data_fold_idx)

eval_dataset = FaceFeatureDataset(subject_under_test=subject_under_test, target_idx=target_idx , test=False,  data_dir='detected/', evaluating=True, use_holdout=False, threshold_percentile=threshold_percentile, balance_data=False, regression=regression, frames_pre=frames_pre, frames_after=frames_after, binary_classification=binary_classification, data_fold_idx=data_fold_idx)

# Data Loader (Input Pipeline)
if regression: 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, drop_last=True)
else: 
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, sampler=train_dataset.sampler, drop_last=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=batch_size)
eval_loader  = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False)

# Initialize Neural net
net = Net(input_size=train_dataset.feature_size, embed_size=embed_size, h_sizes=h_sizes, dropout=dropout, num_classes1=num_classes1, num_classes2=num_classes2, apply_event_mask=apply_event_mask, regression=regression, window_size=window_size)
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
    baseline_loss_val = test_model(0, net, num_classes1, eval_dataset, eval_loader, regression, batch_size, fix_dist=label_distribution, validation=True)

# Train the Model
for epoch in range(num_epochs):
    if epoch % 2 == 0:
        test_loss = test_model(epoch, net, num_classes1, test_dataset, test_loader, regression, batch_size)
        eval_loss = test_model(epoch, net, num_classes1, eval_dataset, eval_loader, regression, batch_size, validation=True)
        # Save the model with lowest test loss at specified dir
        # Will evaluate performance on validation set
        if test_loss < lowest_loss:
            lowest_loss = test_loss
            best_model_wts = copy.deepcopy(net.state_dict())           
    if epoch == num_epochs - 1: break
    if epoch > 0 and epoch % 25 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8

    net.train()
    for i, (img_features, labels, _, event_mask, binary_labels1, binary_labels2, annot_targets) in enumerate(train_loader):  
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        img_features = Variable(img_features.float()).cuda()
        event_mask = Variable(event_mask.float()).cuda()
        labels = Variable(labels).cuda()
        binary_labels1 = Variable(binary_labels1).cuda()
        binary_labels2 = Variable(binary_labels2).cuda()
        annot_targets = Variable(annot_targets).cuda()

        if regression:           
            outputs_mu, outputs_sigma, outputs_rec = net(img_features.float(), event_mask)
            dist = Normal(loc=outputs_mu, scale=outputs_sigma)
            loss1 = torch.mean(-dist.log_prob(labels.float()))
            loss2 = mseLoss(outputs_mu.squeeze(), labels.float())
            loss3 = mseLoss(outputs_rec, annot_targets.float())
            if include_aux_loss: 
                loss = log_PDF_loss_weight*loss1 + loss2 + loss3
            else: 
                loss = log_PDF_loss_weight*loss1 + loss2
        else:
            outputs, outputs_rec = net(img_features.float(), event_mask)
            if target_idx == 1:
                binary_outputs1 = torch.stack((torch.log(torch.sum(torch.exp(outputs[:,0:2]), dim=1)), outputs[:,2]), 1) 
                binary_outputs2 = torch.stack((outputs[:,0], torch.log(torch.sum(torch.exp(outputs[:,1:3]), dim=1))), 1)
            else:
                binary_outputs1 = outputs
                binary_outputs2 = outputs
            
            if binary_loss_option == 1:
                loss2 = ceLoss_train(binary_outputs1, binary_labels1.long())
            elif binary_loss_option == 2:
                loss2 = ceLoss_train(binary_outputs2, binary_labels2.long())
            elif binary_loss_option == 3:
                loss2 = 0.5 * (ceLoss_train(binary_outputs1, binary_labels1.long()) + ceLoss_train(binary_outputs2, binary_labels2.long()))

            loss1 = ceLoss_train(outputs, labels.long())       
            loss3 = mseLoss(outputs_rec, annot_targets.float())
            if include_binary_loss: 
                if include_aux_loss: loss = loss1 + 2*loss2 + loss3
                else: loss = loss1 + 2*loss2
            else: 
                if include_aux_loss: loss = loss1 + loss3
                else: loss = loss1

        loss.backward()
        optimizer.step()
        if (i+1) % 20 == 0:
            if regression:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, LogPDF_Loss: %.4f, MSE_Loss: %.4f, Aux_Loss: %.4f' 
               %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item()))
            else:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Classification_Loss: %.4f, Binary_Classification_Loss: %.4f, Aux_loss: %.4f' 
               %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item(), loss1.data.item(),  loss2.data.item(), loss3.data.item()))

torch.save(best_model_wts, 'MLP_facs_reward_models/'+subject_under_test+'_'+str(np.round(lowest_loss,4))+'.pkl')
