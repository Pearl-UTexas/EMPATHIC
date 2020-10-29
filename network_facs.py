import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size=1024, embed_size=32, h_sizes=[16,32,16,16], window_size=2+24+1, num_classes1=3, num_classes2=98, dropout=0.4, feature_divisions = [512,61,35,54,40,10], apply_event_mask = False, regression = False, include_delta_feature = False):
    #def __init__(self, input_size=1024, embed_size=64, h_sizes=[64,32,32,8], window_size=2+24+1, num_classes1=3, num_classes2=98, dropout=0.4, feature_divisions = [512,61,35,54,40,10], apply_event_mask = False, regression = False, include_delta_feature = False):
        super(Net, self).__init__()
        
        self.input_size = input_size
        self.feature_divisions = feature_divisions # divisions per feature vector for separating inputs
        self.apply_event_mask = apply_event_mask
        self.include_delta_feature = include_delta_feature
        self.h_sizes = h_sizes
        self.regression = regression

        # Encoding layer for VGG-emotion features
        '''
        self.vgg_input = nn.Linear(feature_divisions[0], hidden_size)
        self.lmk_input = nn.Linear(feature_divisions[1], hidden_size)
        '''
        self.facs_input = nn.Linear(feature_divisions[2]*window_size, embed_size)
        input_dim = embed_size
        self.pose_input = nn.Linear(feature_divisions[3]*window_size, embed_size//2)
        input_dim += embed_size//2
        '''
        self.pmd_input = nn.Linear(feature_divisions[4], hidden_size3)
        
        self.detector_input = nn.Linear(feature_divisions[5]*window_size, embed_size)
        '''

        if self.include_delta_feature:
            self.delta_facs_input = nn.Linear(feature_divisions[2]*window_size, embed_size)
            input_dim += embed_size

        if self.apply_event_mask: 
            self.event_mask_fc = nn.Linear(1, embed_size//4)
            input_dim += embed_size//4

        if len(h_sizes) > 0:
            self.hidden = nn.ModuleList()        
            self.hidden.append(nn.Linear(input_dim, h_sizes[0]))
            self.hidden.append(nn.BatchNorm1d(num_features=h_sizes[0]))
            self.hidden.append(nn.LeakyReLU())
            self.hidden.append(nn.Dropout(dropout))
            for k in range(0, len(h_sizes)-1):                
                self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k+1]))
                self.hidden.append(nn.BatchNorm1d(num_features=h_sizes[k+1]))
                self.hidden.append(nn.LeakyReLU())
                self.hidden.append(nn.Dropout(dropout))

            if regression:
                self.out_mu = nn.Linear(h_sizes[-1], 1) 
                self.out_sigma = nn.Linear(h_sizes[-1], 1)
            else:
                self.out = nn.Linear(h_sizes[-1], num_classes1)
            # auxiliary task
            #self.fc30_rec = nn.Linear(h_sizes[0], h_sizes[-1])
            #self.fc31_rec = nn.Linear(h_sizes[-1], num_classes2) 
            self.fc_rec = nn.Linear(h_sizes[0], num_classes2) 
        else: # No hidden layer?? -> only embedding layer (one hidden layer)
            if regression:
                self.out_mu = nn.Linear(input_dim, 1) 
                self.out_sigma = nn.Linear(input_dim, 1)
            else:
                self.out = nn.Linear(input_dim, num_classes1)
            # auxiliary task
            #self.fc30_rec = nn.Linear(input_dim, input_dim)
            #self.fc31_rec = nn.Linear(input_dim, num_classes2) 
            self.fc_rec = nn.Linear(input_dim, num_classes2) 

        
        self.init_weights()

    def init_weights(self):
        '''
        torch.nn.init.kaiming_uniform_(self.vgg_input.weight)
        torch.nn.init.kaiming_uniform_(self.lmk_input.weight)
        torch.nn.init.kaiming_uniform_(self.facs_input.weight)
        torch.nn.init.kaiming_uniform_(self.pose_input.weight)
        torch.nn.init.kaiming_uniform_(self.pmd_input.weight)
        torch.nn.init.kaiming_uniform_(self.event_mask_fc.weight)
        '''
        #torch.nn.init.kaiming_uniform_(self.detector_input.weight)
        for i in range(len(self.hidden)): 
            if isinstance(self.hidden[i], nn.Linear):
                torch.nn.init.kaiming_uniform_(self.hidden[i].weight)
        
        if self.regression:
            torch.nn.init.kaiming_uniform_(self.out_mu.weight)
            torch.nn.init.kaiming_uniform_(self.out_sigma.weight)
        else:
            torch.nn.init.kaiming_uniform_(self.out.weight)

        if self.include_delta_feature:
            torch.nn.init.kaiming_uniform_(self.delta_facs_input.weight)
        #torch.nn.init.kaiming_uniform_(self.fc30_rec.weight)
        #torch.nn.init.kaiming_uniform_(self.fc31_rec.weight)
        torch.nn.init.kaiming_uniform_(self.fc_rec.weight)

    def forward(self, src, mask):
        # Encode VGG features
        feat_idx = 1
        # vgg_out = self.vgg_input(src[:,:,0:sum(self.feature_divisions[0:feat_idx])])    
        # Encode lmk_dist features
        feat_idx += 1
        # lmk_out = self.lmk_input(src[:,:,sum(self.feature_divisions[0:feat_idx-1]):sum(self.feature_divisions[0:feat_idx])])
        # Encode facs features
        feat_idx += 1
        input_feat = src[:,:,sum(self.feature_divisions[0:feat_idx-1]):sum(self.feature_divisions[0:feat_idx])]

        '''
        input_feat = src[:,:,sum(self.feature_divisions[0:feat_idx-1]):sum(self.feature_divisions[0:feat_idx])]
        enc_shape = input_feat.shape
        if self.include_delta_feature:
            facs_in = input_feat[:, :enc_shape[1]//2, :].reshape(enc_shape[0],enc_shape[1]*enc_shape[2]//2)
            delta_facs_in = input_feat[:, enc_shape[1]//2:, :].reshape(enc_shape[0],enc_shape[1]*enc_shape[2]//2)
            delta_facs_out = self.delta_facs_input(delta_facs_in)
        else:
            facs_in = input_feat.reshape(enc_shape[0],enc_shape[1]*enc_shape[2])
        facs_out = self.facs_input(facs_in)
        '''
        # Encode pose features
        feat_idx += 1
        pose_feat = src[:,:,sum(self.feature_divisions[0:feat_idx-1]):sum(self.feature_divisions[0:feat_idx])]
        pose_shape = pose_feat.shape
        pose_out = self.pose_input(pose_feat.reshape(pose_shape[0],pose_shape[1]*pose_shape[2]))
        # Encode pmd features
        feat_idx += 1
        # pmd_out = self.pmd_input(src[:,:,sum(self.feature_divisions[0:feat_idx-1]):])
        # Encode detector features
        feat_idx += 1
        #input_feat = src[:,:,sum(self.feature_divisions[0:feat_idx-1]):sum(self.feature_divisions[0:feat_idx])]

        # Reshape features into batch_size*dim
        enc_shape = input_feat.shape
        if self.include_delta_feature:
            facs_in = input_feat[:, :enc_shape[1]//2, :].reshape(enc_shape[0],enc_shape[1]*enc_shape[2]//2)
            delta_facs_in = input_feat[:, enc_shape[1]//2:, :].reshape(enc_shape[0],enc_shape[1]*enc_shape[2]//2)
            delta_facs_out = self.delta_facs_input(torch.clamp(delta_facs_in, -1.0, 1.0))
        else:
            facs_in = input_feat.reshape(enc_shape[0],enc_shape[1]*enc_shape[2])
              
        const_c = Variable(torch.Tensor([0.5]).float())
        facs_out = self.facs_input(torch.clamp(facs_in,0.0,1.0) - const_c.expand(facs_in.size()).cuda())

        if self.apply_event_mask:
            event_mask = self.event_mask_fc(mask)
            if self.include_delta_feature:
                output = torch.cat((facs_out, delta_facs_out, pose_out, event_mask),dim=1)
            else:
                #output = torch.cat((facs_out, detector_out, event_mask),dim=1)
                output = torch.cat((facs_out, pose_out, event_mask),dim=1)
        else:
            if self.include_delta_feature:
                output = torch.cat((facs_out, delta_facs_out),dim=1)
            else:
                output = torch.cat((facs_out, pose_out),dim=1)
                #output = facs_out

        # output = torch.cat((vgg_out,lmk_out,facs_out,pose_out,pmd_out),dim=1)
        h_ct = 0
        for h in self.hidden: 
            output = h(output)
            #if h_ct == 0: output_rec = self.fc30_rec(output)
            if h_ct == 1: output_rec = self.fc_rec(output)
            h_ct += 1
        if self.regression:
            out_mu = self.out_mu(output) # output for advantage
            out_sigma = torch.abs(self.out_sigma(output)) # output for advantage
            return (out_mu, out_sigma, None)
        else:
            out = self.out(output)
            #output_rec = self.fc31_rec(output_rec) 
            return (out, output_rec)
        

if __name__ == '__main__':
    net = Net()
    print(net)
