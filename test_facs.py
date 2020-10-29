import sys
import os
import random
from os import path

from os.path import isfile, isdir, join

import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from data_loader_facs import FaceFeatureDataset
from network_facs import Net

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.spatial import distance


batch_size = 1

target_idx = 1 # training target index
subject_under_test= sys.argv[1] #'tcJ5dqCJkz'

model_dir = './MLP_detector_reward_models/' 
all_models = [join(model_dir, f) for f in os.listdir(model_dir) if isfile(join(model_dir, f))]
best_model = None
lowest_loss = 100.0

for model_path in all_models:
    model_info = model_path.replace('.pkl', '')
    model_info = model_info.split('/')[-1].split('_')

    if not model_info[0] == subject_under_test: continue
    loss = float(model_info[-1])
    model_threshold_percentile = int(model_info[-2])
    if model_threshold_percentile != 0: continue
    model_type = model_info[-3]
    if model_type != 'facsposeauxbinary': continue
    if loss < lowest_loss:
        lowest_loss = loss
        best_model = model_path

threshold_percentile = 0

if best_model is None: 
    print('No model found for', subject_under_test, 'in', model_dir)
    sys.exit(0)


frames_pre = 4
frames_after = 18 #24
window_size = frames_pre + frames_after + 1
num_classes1=3
num_classes2=window_size * 10
 
celoss = nn.CrossEntropyLoss()

test_dist = (0.8,0.1,0.1)


test_dataset = FaceFeatureDataset(subject_under_test=subject_under_test, target_idx=target_idx , test=False,  data_dir='detected/', evaluating=True, use_holdout=False, threshold_percentile=threshold_percentile, balance_data=False, frames_pre=frames_pre, frames_after=frames_after)
#test_dataset = FaceFeatureDataset(subject_under_test=subject_under_test, target_idx=target_idx , test=False,  data_dir='detected/', evaluating=False, use_holdout=False, threshold_percentile=threshold_percentile, balance_data=False)

# Data Loader (Input Pipeline)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
#net = Net(input_size=test_dataset.feature_size, num_classes1=num_classes1, num_classes2=num_classes2, apply_event_mask=False) 
net = Net(input_size=test_dataset.feature_size, num_classes1=num_classes1, num_classes2=num_classes2, apply_event_mask=False, window_size=window_size)

total = 0


classes = [-5,-1,6,0]
all_matches = {}

for i in range(len(classes)-1):
    for j in range(len(classes)-1):
        if i == j: continue
        for k in range(len(classes)-1):
            if k == i or k == j: continue
            all_matches[(i,j,k)] = []

testcase_losses = []

testcase_matches = copy.deepcopy(all_matches)
print('loading model:',best_model)
net.load_state_dict(torch.load(best_model))
for param in net.parameters():
    net.requires_grad = False
    
net.eval()
net.cuda()
confusion_matrix = torch.zeros(num_classes1, num_classes1)
correct = 0
total = 0
total_loss = 0

output_fh = open('evaluation_outputs/'+subject_under_test+'.csv','w')

for i,  (img_features, labels, _, event_mask, _, _) in enumerate(test_loader):
    img_features = Variable(img_features.float()).cuda()
    event_mask = Variable(event_mask.float()).cuda()
    labels = Variable(labels).cuda()
    outputs, _ = net(img_features.float(), event_mask)
    loss = celoss(outputs, labels.long())
    testcase_losses.append(loss.item())
    outputs = outputs.cpu().data.float()                
    total_loss += loss.data.item()            
    labels = labels.float().view(-1)                
    _, predicted = torch.max(torch.Tensor(outputs), 1)
    predictions = predicted.cpu().view(-1)               
    for t, p in zip(labels.view(-1), predicted.cpu().view(-1)):
        confusion_matrix[t.long(), p.long()] += 1
    total += labels.size(0)       
    correct += (predicted.cpu() == labels.cpu()).sum()  
    print(i, outputs)
    output_fh.write(str(i)+','+','.join([str(num.data.item()) for num in outputs[0]])+'\n')
    r_levels = test_dataset.reaction_levels[i*batch_size:(i+1)*batch_size]
    for idx in range(len(labels)):
        label = int(labels[idx].item())
        if label != 3: # condition on pickup         
            output = outputs[idx]                 
            for matching in testcase_matches: # all_matches:
                testcase_matches[matching].append(output[matching[label]].item()) 
                #testcase_matches[matching].append(output[matching[label]].item() * r_levels[idx]) 
                #testcase_matches[matching].append(1.0) #test_dist[matching[label]]) 
   
print('acc: ', correct*100.0/total)
print('total loss: ', total_loss/len(test_dataset))
print(confusion_matrix)
for cls in range(confusion_matrix.shape[0]):
    total_cls = torch.sum(confusion_matrix[cls]).data.item()
    if total_cls != 0:
        accuracy = confusion_matrix[cls, cls].data.item()*1.0 / total_cls
        print('class', cls, confusion_matrix[cls,cls].data.item(),'/',total_cls, ', acc:', accuracy*100.0,'%')


num_pickup_pivots = [25, 50, 75, 125] #[*range(1,240)]
total_score = [0.0]*(len(num_pickup_pivots)+1) #  intervals
   
scores = []
print(subject_under_test ,' CE loss:', np.mean(testcase_losses))
score_idx = 0
pivots = copy.deepcopy(num_pickup_pivots)

pivots.append(len(testcase_matches[(0,1,2)]))

for num_pickups in pivots:       
    if num_pickups == num_pickup_pivots[0]:
        for matching in testcase_matches:
            testcase_matches[matching] = np.array(testcase_matches[matching])
            testcase_matches[matching] /= 100.0
    logits = {}
    
    matching_keys = list(all_matches.keys())
    random.shuffle(matching_keys)
    for matching in matching_keys:
        logits[matching] = np.sum(testcase_matches[matching][0:num_pickups])
    exp_sum = sum([np.exp(logit) for logit in logits.values()])        
    
    testcase_matches_probs = {}
    matching_keys = list(testcase_matches.keys())
    random.shuffle(matching_keys)
    for matching in matching_keys:
        testcase_matches_probs[matching] = np.exp(logits[matching])/exp_sum     
               
    #print(testcase_matches_probs)
    matching = max(testcase_matches_probs.items(), key=operator.itemgetter(1))[0]
    
    score = 0
    for i in range(len(matching)):
        if i == matching[i]: score += 1
    if score == 1 and (matching[0] == 1 and matching[1] == 0 or matching[1] == 2 and matching[2] == 1):
        score += 1
    total_score[score_idx] += score
    score_idx += 1
    print('[num pred frames ',num_pickups, '] Best guess:', matching, ' Score:', score)
    scores.append(score)

print('Final score,', ','.join([str(score) for score in scores]))
#plt.plot(scores)
#plt.show()
    



