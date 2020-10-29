import os
from os import path
from os.path import isfile, isdir, join
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import imageio



# dataset meta info

data_files = {
               2: 'detected/RLstatistics_episode_2.csv',
              30: 'detected/RLstatistics_episode_30.csv',           
              31: 'detected/RLstatistics_episode_31.csv',           
              37: 'detected/RLstatistics_episode_37.csv',   
              41: 'detected/RLstatistics_episode_41.csv',
              46: 'detected/RLstatistics_episode_46.csv'
              }
              
# conditions
condition_dic = {
    "NUmTnWWjX6":0,    "4bzq92vQGe":0,    "lxJtNRlUAs":0,    "J3sGDFKNx5":0,
    "pKH0e6bBIo":0,    "G4fUr3vTFO":0,    "Oz4PI7OLOi":0,    "YBVcCZS7fy":0,
    "6cMbVYpG00":0,    "fYWPkRcKi1":0,    "BDRiXPwtNf":0,    "WkOsToXr9v":0,
    "S6Zghgggo4":0,    "Kpd16ANmf3":0,    "SoQ2uxHSHw":0,    "oI7RlzkU2k":0,
    "2iA4jV97rl":0,    "J4LTGkh8bC":1,    "tcJ5dqCJkz":1,    "eneTslHMcV":1,
    "Z2pZR2LQxg":1,    "3gCI2rjrdp":1,    "B9ASGFKtcI":1,    "GHiCGHgING":1,
    "5uKjBzbCUY":1,    "fNzfYRiG2Q":1,    "8EwwxdFc3G":1,    "1JIuegvQCL":1,
    "eBIJoBu3xw":1,    "NvWqZhMTIr":1,    "Gm2fspTh0O":1,    "SYf80nuPOV":1,
    "gSSEPyUoeT":1,    "xNWYOcsU9X":1,    "2Lk7llB0CT":1,    "bo3OrtQR5v":1,
    "qU8Frq52yW":1
                }



unconditioned = [
    "NUmTnWWjX6",    "4bzq92vQGe",    "lxJtNRlUAs",    "J3sGDFKNx5",
    "pKH0e6bBIo",    "G4fUr3vTFO",    "Oz4PI7OLOi",    "YBVcCZS7fy",
    "6cMbVYpG00",    "fYWPkRcKi1",    "BDRiXPwtNf",    "WkOsToXr9v",
    "S6Zghgggo4",    "Kpd16ANmf3",    "SoQ2uxHSHw",    "oI7RlzkU2k",
    "2iA4jV97rl"
                 ]




def load_feature_data(test_subj, detected_dir = './detected', feature='detectionFrames', condition=0, test=False, evaluating=False, use_holdout=False, filter_subject=False, finetune=False, subj_reaction_level_threshold=0.0, data_fold_idx=0):

    subj_idxed_feature_data = {}
    feature_dirs = [join(detected_dir, f) for f in os.listdir(detected_dir) if isdir(join(detected_dir, f))]
    subj_w_annotations = unconditioned 
    
    for feature_dir in feature_dirs:
        subj_id = feature_dir.strip().split('/')[-1]
        if not subj_id in subj_w_annotations: continue
        if condition_dic[subj_id] != condition: continue
        if evaluating and subj_id != test_subj: continue
        if use_holdout and subj_id != test_subj: continue

        # compute detection frame indexes
        hash_num = sum([ord(test_subj[ch_idx]) for ch_idx in range(len(test_subj))]) + sum([ord(subj_id[ch_idx]) for ch_idx in range(len(subj_id))])
        hash_test_idx = (hash_num + data_fold_idx) % 4
        subj_train_idx = (hash_num + data_fold_idx + 1) % 4 
        #print(subj_id, hash_test_idx, subj_train_idx)
        
        subj_idxed_feature_data[subj_id] = {}
        # list all the episodes under the file
        feature_files = [join(feature_dir, f) for f in os.listdir(feature_dir) if isfile(join(feature_dir, f))]
        for feature_file in feature_files:
            # Looking for subjid_episode_n_detectionFrames_k.csv 
            info = feature_file.strip().split('/')[-1].split('.')[0].split('_')
            if (not use_holdout) and info[-1] == 'holdout': continue # let ALONE the holdout set!
            if info[-2] != feature:  continue # check file postfix
            if info[1] != 'episode': continue # check file category
            # -> test: k = 0, train: k = 1,2,3
            if use_holdout and info[-1] != 'holdout': continue
            elif not use_holdout:
                k = int(info[-1])
                if evaluating:
                    if k == hash_test_idx or k == subj_train_idx: continue
                elif test: 
                    if k != hash_test_idx: continue # test skip other than hash_test_idx
                else: # train
                    if subj_id == test_subj and k != subj_train_idx: continue # train
                    elif k == hash_test_idx: continue
            else:
                k = 0

            episode_num = int(info[2]) 
            if not episode_num in subj_idxed_feature_data[subj_id]:
                subj_idxed_feature_data[subj_id][episode_num] = {}
            subj_idxed_feature_data[subj_id][episode_num][k] = pd.read_csv(feature_file, header=None)
    return subj_idxed_feature_data
    
    
class FaceFeatureDataset(Dataset):
    """Face Feature dataset."""

    def __init__(self, subject_under_test='5uKjBzbCUY', data_dir='detected/', target_idx=1, regression=False, 
                       test=False,  evaluating=False, use_holdout=False,
                       threshold_percentile=0,
                       balance_data=True,
                       filter_subject=False, subj_reaction_level_threshold=0.20,
                       frames_pre=2, frames_after=24,
                       binary_classification=False,
                       include_delta_feature = False,
                       finetune=False,
                       data_fold_idx=0
                 ):
       
        """
        Args:
            subject_under_test: the subject will be evaluated on, use the
                                rest of other subjects under the same condition for train + test. 
            data_dir:           root directory saving all data files  
        """
        
        self.subject_under_test = subject_under_test
        self.subj_idxed_feature_frames = load_feature_data(subject_under_test, detected_dir=data_dir, condition=condition_dic[subject_under_test], test=test, evaluating=evaluating, use_holdout=use_holdout, filter_subject=filter_subject, subj_reaction_level_threshold=subj_reaction_level_threshold, finetune=finetune, data_fold_idx=data_fold_idx)
        self.feature_start_index = 2        
        self.feature_size = 512 + 61 + 35 + 54 + 40 + 10
        self.feature_end_index = self.feature_start_index + self.feature_size
        self.threshold_percentile = threshold_percentile
        self.include_delta_feature = include_delta_feature
        self.filter_subject = filter_subject
        
        self.targets_frames = {}
        for data_file_id in data_files:
            self.targets_frames[data_file_id] = pd.read_csv(data_files[data_file_id])
        self.target_idx = target_idx # timestep 1.reward 2.optimality 3.qval_opt 4.qval_beh 5.adv_opt 6.adv_beh 7.suprise r(t-1) t(t-2)
        self.binary_classification = binary_classification

        if self.target_idx == 1:  
            if self.binary_classification:
                self.weight_mappings = {-5:0, -1:0, 6:1, 0:2}
            else:
                self.weight_mappings =  {-5:0, -1:1, 6:2, 0:3}
            self.binary_weight_mappings1 = {-5:0, -1:0, 6:1, 0:2}
            self.binary_weight_mappings2 = {-5:0, -1:1, 6:1, 0:2}
        elif self.target_idx == 2: 
            self.weight_mappings =  {0:0, 1:1}
                
        self.frames_pre = frames_pre
        self.frames_after = frames_after
        self.window_size = self.frames_pre + self.frames_after + 1
        self.class_weights = list(np.ones(3)) if not self.binary_classification else list(np.ones(2))
        self.binary_class_weights = list(np.ones(2))
        self.threshold = 0.0
       
        self.generate_training_data()
        
        self.label_distribution = None
        if balance_data and not regression:  
            self.balance_data_by_class()
        
        
    def generate_training_data(self):
    
        self.data_size = 0  
        self.processed_training_data = []
        self.targets_data = []
        self.reaction_levels = []
        self.event_masks = []
        self.subj_ids = []
        self.annot_targets = []
        
        # preprocess data
        for subj_id in self.subj_idxed_feature_frames:
            for episode_num in self.subj_idxed_feature_frames[subj_id]:
                targets_frame = self.targets_frames[episode_num]
                prev_img_feature = None
                for k in self.subj_idxed_feature_frames[subj_id][episode_num]:
                    curr_dataframe = self.subj_idxed_feature_frames[subj_id][episode_num][k]
                    curr_data_size = len(curr_dataframe)
                    self.data_size += curr_data_size  
                    for idx in range(curr_data_size):
                        data_idx = []   
                        for d_idx in range(idx-self.frames_pre, idx+self.frames_after+1):
                            if d_idx < 0: continue
                            if d_idx >= curr_data_size: continue
                            data_idx.append(d_idx)                        
                        while len(data_idx) < self.window_size:
                            for i in range(data_idx[0]-(idx-self.frames_pre)): 
                                data_idx.insert(0, data_idx[0])
                            for i in range(idx + self.frames_after - data_idx[-1]): 
                                data_idx.append(data_idx[-1])

                        img_feature = np.array(curr_dataframe.iloc[data_idx, self.feature_start_index:self.feature_end_index]).astype('float')
                        #for a_i in range(512+61+35,512+61+35+54): img_feature[:,a_i] = 0 
                        annot_target =  np.array(curr_dataframe.iloc[data_idx, self.feature_end_index-10:self.feature_end_index]).astype('float')
                        event_mask = []
                        frame_num = int(curr_dataframe.iloc[idx, 0])
                        #if targets_frame.iloc[int(curr_dataframe.iloc[idx, 0]), 1] != 0: event_mask.append(1.0) # Whether reward==0
                        #else:  event_mask.append(0.0)
                        event_mask.append(frame_num//50)

                        mean_max_reaction_level = np.mean(np.amax(np.clip(img_feature[:,512+61:512+61+35],0.0,1.0), axis=1))

                        if prev_img_feature is None:
                            delta_feature = img_feature
                        else:
                            delta_feature = img_feature - prev_img_feature
                        prev_img_feature = img_feature

                        if self.include_delta_feature:
                            self.processed_training_data.append(np.concatenate((img_feature, delta_feature), axis=0))
                        else:
                            #self.processed_training_data.append(delta_feature) 
                            self.processed_training_data.append(img_feature)
                        self.annot_targets.append(annot_target.flatten())
                        tar_idx = int(curr_dataframe.iloc[idx, 0])
                        #print(tar_idx)
                        self.targets_data.append(int(targets_frame.iloc[tar_idx, self.target_idx]))
                        self.reaction_levels.append(mean_max_reaction_level)
                        self.event_masks.append(np.array(event_mask))
                        self.subj_ids.append(subj_id)
                                    
        # Convert to np array for better indexing
        self.processed_training_data = np.array(self.processed_training_data)
        self.targets_data = np.array(self.targets_data)
        self.reaction_levels = np.array(self.reaction_levels)
        self.event_masks = np.array(self.event_masks)
        self.subj_ids = np.array(self.subj_ids)
        self.annot_targets = np.array(self.annot_targets)

        # Filter classes
        sampled_indices = np.squeeze(np.argwhere(abs(self.targets_data) > 0))

        self.processed_training_data = self.processed_training_data[sampled_indices]
        self.targets_data = self.targets_data[sampled_indices]
        self.reaction_levels = self.reaction_levels[sampled_indices]
        self.event_masks = self.event_masks[sampled_indices]
        self.subj_ids = self.subj_ids[sampled_indices]
        self.annot_targets = self.annot_targets[sampled_indices]

        # Get the threshold by percentile
        if self.threshold_percentile > 0:
            self.threshold = 0.5 #threshold
            sampled_indices = np.squeeze(np.argwhere(self.reaction_levels > self.threshold))

            # Sample by thresholded indices
            self.processed_training_data = self.processed_training_data[sampled_indices]
            self.targets_data = self.targets_data[sampled_indices]
            self.reaction_levels = self.reaction_levels[sampled_indices]
            self.event_masks = self.event_masks[sampled_indices]
            self.subj_ids = self.subj_ids[sampled_indices]
            self.annot_targets = self.annot_targets[sampled_indices]

        # self.data_size = len(self.processed_training_data)
        self.data_size = int(self.processed_training_data.shape[0])#(1-0.01*self.threshold_percentile)*self.data_size)


    def balance_data_by_class(self):
        labels = [self.weight_mappings[target] for target in self.targets_data]
        counts, _ = np.histogram(labels, bins=range(len(self.class_weights)+1))
        self.label_distribution = np.divide(counts,sum(counts))
        class_weights = np.divide(1.0/len(self.weight_mappings), np.divide(counts,sum(counts)))
        class_weights = np.divide(class_weights, sum(class_weights))
        self.update_sample_weights(class_weights)


    def update_sample_weights(self, class_weights):
        self.class_weights = class_weights
        self.sample_weights = []
        for data_sample_ct in range(len(self.targets_data)):
            subj_id = self.subj_ids[data_sample_ct]
            if subj_id == self.subject_under_test:
                self.sample_weights.append(self.class_weights[self.weight_mappings[self.targets_data[data_sample_ct]]]*1.5)
            else:
                self.sample_weights.append(self.class_weights[self.weight_mappings[self.targets_data[data_sample_ct]]])
        self.sample_weights = torch.DoubleTensor(self.sample_weights)
        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(self.sample_weights, len(self.sample_weights))

        
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        """
        Logic for loading one training instance.
        """
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_feature = self.processed_training_data[idx]
        event_mask = self.event_masks[idx]        
        target = self.targets_data[idx]
        subj_id = self.subj_ids[idx]
        binary_labels1 = None
        binary_labels2 = None
        annot_target = self.annot_targets[idx]

        if self.target_idx == 1 or self.target_idx == 2: 
            label = np.array(self.weight_mappings[target]).astype('long')
            if self.target_idx == 1:
                binary_labels1 = np.array(self.binary_weight_mappings1[target]).astype('long')
                binary_labels2 = np.array(self.binary_weight_mappings2[target]).astype('long')
            else:
                binary_labels1 = np.array(target).astype('long')
                binary_labels2 = np.array(target).astype('long')
        else: 
            label = np.array(target)
            binary_labels1 = np.array(target) # just a placeholder
            binary_labels2 = np.array(target) # just a placeholder

        return (img_feature, label, subj_id, event_mask, binary_labels1, binary_labels2, annot_target)    
    


if __name__ == '__main__':
    import torch.nn as nn
    from torch.autograd import Variable
    criterion = nn.NLLLoss()
    dataset_num = 1
    
    test_dataset = FaceFeatureDataset(subject_under_test='G4fUr3vTFO', data_dir='detected/', test=True)

    batch_size=16
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
           batch_size=batch_size, sampler=test_dataset.sampler)    

    for (img_features, labels, _, event_mask, binary_labels1, binary_labels2, annot_target) in test_loader:
        print(img_features.shape, labels.shape, binary_labels1.shape, binary_labels2.shape, annot_target.shape)
        print(annot_target[0][0:11])
 
        
