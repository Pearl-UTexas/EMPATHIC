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

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def ranking_loss(output, target): # batched ranking loss
    loss_criterion = nn.CrossEntropyLoss()
    target_onehot = torch.Tensor(to_categorical(target.cpu(), 3)).cuda()
    reward_values = torch.Tensor([-5.0,-1.0,6.0])
    reward_values = reward_values.cuda()
    pred_rewards = torch.mv(output, reward_values)
    real_rewards = torch.mv(target_onehot, reward_values)
    batch_size = real_rewards.shape[0]
    # import pdb;pdb.set_trace()
    model_ranking_logits = torch.cat((torch.sum(pred_rewards[0:batch_size//2], dim=0, keepdim=True),torch.sum(pred_rewards[batch_size//2:], dim=0, keepdim=True)),0).unsqueeze(0)
    if torch.sum(real_rewards[0:batch_size//2]) > torch.sum(real_rewards[batch_size//2:]): ranking_label = torch.Tensor([1]).cuda()
    else: ranking_label = torch.Tensor([0]).cuda()
    loss = loss_criterion(model_ranking_logits, ranking_label.long())
    return loss

def test_model(epoch, net, num_classes1, dataset, data_loader, regression, batch_size, validation=False, fix_dist=None):
    # Test the Model
    set_name = 'test'
    if validation: set_name = 'validation'
    net.eval()
    confusion_matrix = torch.zeros(num_classes1, num_classes1)
    correct = 0
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

        if regression:           
            outputs_mu, outputs_sigma, outputs_rec = net(img_features.float(), event_mask)
            dist = Normal(loc=outputs_mu, scale=outputs_sigma)
            loss1 = 0.005*torch.mean(-dist.log_prob(labels.float()))
            loss2 = mseLoss(outputs_mu.squeeze(), labels.float())
            # loss3 = mseLoss(outputs_rec.squeeze(), vgg_targets.float())
        else:
            if not fix_dist is None:
                outputs = np.tile(np.log(fix_dist), labels.shape[0]).reshape(labels.shape[0], -1)
                outputs = torch.Tensor(outputs).cuda().float()
            else:
                outputs, annot_pred = net(img_features.float(), event_mask)                      
                loss3 = mseLoss(annot_pred.squeeze(), annot_target.float())
            binary_outputs1 = torch.stack((torch.log(torch.sum(torch.exp(outputs[:,0:2]), dim=1)), outputs[:,2]), 1) 
            binary_outputs2 = torch.stack((outputs[:,0], torch.log(torch.sum(torch.exp(outputs[:,1:3]), dim=1))), 1) 
            
            if binary_loss_option == 1:
                loss2 = ceLoss_test(binary_outputs1, binary_labels1.long())
            elif binary_loss_option == 2:
                loss2 = ceLoss_test(binary_outputs2, binary_labels2.long())
            elif binary_loss_option == 3:
                loss2 = 0.5 * (ceLoss_test(binary_outputs1, binary_labels1.long()) + ceLoss_test(binary_outputs2, binary_labels2.long()))

            loss1 = ceLoss_test(outputs, labels.long())
            _, predicted = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), predicted.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += (predicted.cpu() == labels.cpu()).sum()   
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
            # avg_loss3 = float(total_loss3)/(float(dataset.data_size)/batch_size)
            # print('Reconstruction loss on test set: %f' % avg_loss3)
        else:
            print('Log PDF Loss on test set: %f' % avg_loss1)
            print('MSE on test set: %f' % avg_loss2)
            # avg_loss3 = float(total_loss3)/(float(dataset.data_size)/batch_size)
            # print('Reconstruction loss on test set: %f' % avg_loss3)
    else:
        model = 'network'
        if not fix_dist is None: model = 'fix_distribution'
        print('Accuracy of ' + model + ' on ' + set_name + ' images: %d %%' % (100 * correct / total))
        print('Cross Entropy Loss (Reward) of ' + model + ' on ' + set_name + ' set: %f' % avg_loss1)
        print('Cross Entropy Loss (Reward) of binary ' + model + ' on ' + set_name + ' set: %f' % avg_loss2)
        print('MSE Loss (Reward) of ' + model + ' on ' + set_name + ' set: %f' % avg_loss3)
        # print('Reconstruction loss on test set: %f' % avg_loss2)
        print(confusion_matrix)
        for cls in range(confusion_matrix.shape[0]):
            total_cls = torch.sum(confusion_matrix[cls]).data.item()
            if total_cls != 0:
                accuracy = confusion_matrix[cls, cls].data.item()*1.0 / total_cls
                print('eval_class', cls, confusion_matrix[cls,cls].data.item(),'/',total_cls, ', acc:', accuracy*100.0,'%')


    if regression:
        return avg_loss2
    else:
        return avg_loss1

# Hyper Parameters 
num_epochs = 100
batch_size = 8
h_sizes=[16,32,16,16]
embed_size=32
dropout=0.4

frames_pre=4
frames_after=18
window_size = frames_pre + frames_after + 1

apply_event_mask = False
include_binary_loss = True
include_aux_loss = True
binary_classification=False

threshold_percentile = 0
# 1. lumping -5/-1; 2. lumping -1/+6; 3. lumping both in 2 binary losses
binary_loss_option = 1

# timestep 1.reward 2.optimality 3.qval_opt 4.qval_beh 5.adv_opt 6.adv_beh 7.suprise r(t-1) t(t-2)
target_idx = 1 # training target index
if target_idx < 3: regression = False
else: regression = True
learning_rate = 0.002


subject_under_test = sys.argv[1] 
num_classes1 = 3 if not binary_classification else 2
num_classes2 = 10 * window_size

# Loss and Optimizer
#weights = list(np.ones(num_classes1))
#class_weights_train = torch.FloatTensor(weights).cuda()
ceLoss_train = nn.CrossEntropyLoss()
ceLoss_test = nn.CrossEntropyLoss()

mseLoss = nn.MSELoss()



print('threshold_percentile:',threshold_percentile)

train_dataset = FaceFeatureDataset(subject_under_test=subject_under_test, target_idx=target_idx,  data_dir='detected/', threshold_percentile=threshold_percentile, regression=regression, frames_pre=frames_pre, frames_after=frames_after, binary_classification=binary_classification)

test_dataset = FaceFeatureDataset(subject_under_test=subject_under_test, target_idx=target_idx , test=True,  data_dir='detected/', threshold_percentile=threshold_percentile, balance_data=False, regression=regression, frames_pre=frames_pre, frames_after=frames_after, binary_classification=binary_classification)

eval_dataset = FaceFeatureDataset(subject_under_test=subject_under_test, target_idx=target_idx , test=False,  data_dir='detected/', evaluating=True, use_holdout=False, threshold_percentile=threshold_percentile, balance_data=False, regression=regression, frames_pre=frames_pre, frames_after=frames_after, binary_classification=binary_classification)

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
        # Save the Model at specified dir
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
        optimizer.zero_grad()  # zero the gradient buffer
        img_features = Variable(img_features.float()).cuda()
        event_mask = Variable(event_mask.float()).cuda()
        labels = Variable(labels).cuda()
        binary_labels1 = Variable(binary_labels1).cuda()
        binary_labels2 = Variable(binary_labels2).cuda()
        annot_targets = Variable(annot_targets).cuda()
        if regression:           
            outputs_mu, outputs_sigma, outputs_rec = net(img_features.float(), event_mask)
            dist = Normal(loc=outputs_mu, scale=outputs_sigma)
            loss1 = 0.005*torch.mean(-dist.log_prob(labels.float()))
            loss2 = mseLoss(outputs_mu.squeeze(), labels.float())
            loss = loss1 + loss2
        else:
            outputs, outputs_rec = net(img_features.float(), event_mask)           
            #binary_outputs = torch.stack((torch.max(outputs[:,0:2], 1)[0], outputs[:,2]), 1) 
            binary_outputs1 = torch.stack((torch.log(torch.sum(torch.exp(outputs[:,0:2]), dim=1)), outputs[:,2]), 1) 
            binary_outputs2 = torch.stack((outputs[:,0], torch.log(torch.sum(torch.exp(outputs[:,1:3]), dim=1))), 1)  
            
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
                 print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, LogPDF_Loss: %.4f, MSE_Loss: %.4f' 
               %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item(), loss1.data.item(), loss2.data.item()))
            else:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Classification_Loss: %.4f, Binary_Classification_Loss: %.4f, MSE_loss: %.4f' 
               %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data.item(), loss1.data.item(),  loss2.data.item(), loss3.data.item()))

torch.save(best_model_wts, 'MLP_detector_reward_models/'+subject_under_test+'_facsposeauxbinary_'+str(np.round(lowest_loss,4))+'.pkl')
