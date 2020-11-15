import os
from os import path
from os.path import isfile, isdir, join
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import imageio

# Dataset meta info
data_files = {
               2: 'detected/RLstatistics_episode_2.csv',
              30: 'detected/RLstatistics_episode_30.csv',           
              31: 'detected/RLstatistics_episode_31.csv',           
              37: 'detected/RLstatistics_episode_37.csv',   
              41: 'detected/RLstatistics_episode_41.csv',
              46: 'detected/RLstatistics_episode_46.csv'
              }
              
# Conditions of experiment on each human subject
# Some of the subjects in pilot study are not included in our formal experiments
# Refer to the directory detected/ to see the human subjects actually used for training
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

# The human subjects that are included in training our model
unconditioned = [
    "NUmTnWWjX6",    "4bzq92vQGe",    "lxJtNRlUAs",    "J3sGDFKNx5",
    "pKH0e6bBIo",    "G4fUr3vTFO",    "Oz4PI7OLOi",    "YBVcCZS7fy",
    "6cMbVYpG00",    "fYWPkRcKi1",    "BDRiXPwtNf",    "WkOsToXr9v",
    "S6Zghgggo4",    "Kpd16ANmf3",    "SoQ2uxHSHw",    "oI7RlzkU2k",
    "2iA4jV97rl"
                 ]


# Load the data required for learning from the csv files under detected/
def load_feature_data(test_subj, detected_dir = './detected', feature='detectionFrames', condition=0, test=False, evaluating=False, use_holdout=False, data_fold_idx=0):

    print('Restoring image info from ', detected_dir)
    subj_idxed_feature_data = {}
    feature_dirs = [join(detected_dir, f) for f in os.listdir(detected_dir) if isdir(join(detected_dir, f))]
    subj_w_annotations = unconditioned 
    
    for feature_dir in feature_dirs:
        subj_id = feature_dir.strip().split('/')[-1]
        if not subj_id in subj_w_annotations: continue
        if condition_dic[subj_id] != condition: continue
        if evaluating and subj_id != test_subj: continue
        if use_holdout and subj_id != test_subj: continue

        # compute aggregated frame indexes
        hash_num = sum([ord(test_subj[ch_idx]) for ch_idx in range(len(test_subj))]) + sum([ord(subj_id[ch_idx]) for ch_idx in range(len(subj_id))])
        hash_test_idx = (hash_num + data_fold_idx) % 4
        subj_train_idx = (hash_num + data_fold_idx + 1) % 4 
        
        subj_idxed_feature_data[subj_id] = {}
        # list all the episodes under the file
        feature_files = [join(feature_dir, f) for f in os.listdir(feature_dir) if isfile(join(feature_dir, f))]
        for feature_file in feature_files:
            # Looking for subjid_episode_n_detectionFrames_k.csv 
            info = feature_file.strip().split('/')[-1].split('.')[0].split('_')
            if (not use_holdout) and info[-1] == 'holdout': continue # let ALONE the holdout set!
            if info[-2] != feature:  continue # check file postfix
            if info[1] != 'episode': continue # check file category
            # -> testing set: k = 0; training set: k = 1,2,3
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

    def __init__(self, subject_under_test='WkOsToXr9v', data_dir='detected/', target_idx=1, regression=False, 
                       test=False, evaluating=False, use_holdout=False,
                       threshold_percentile=0,
                       balance_data=True,
                       frames_pre=2, frames_after=24,
                       include_delta_feature = False,
                       data_fold_idx=0
                 ):
       
        """
        Args:
            subject_under_test: the subject will be evaluated on, use the rest of other subjects for train + test. 
            data_dir:           root directory saving all data files
            target_idx:         the index of the target statistic to learn and predict.
            regression:         whether to predict other continuous tasks statistics (Q-values, advantage) or not; Currently should always be False (reward only).
            test:               whether to load the test set.
            evaluating:         whether to load the validation set.
            use_holdout:        whether to load the holdout set.
            threshold_precentile: the threshold for sampling subjects with high reaction level; Currently not used.
            balance_data:       whether to balance the data instances based on class distribution.
            frames_pre:         the number of aggregated frames before the current one in the window.
            frames_after:       the number of aggregated frames after the current one in the window.
            include_delta_feature: whether to inlclude delta feature w.r.t the previous frame as an additional feature. Currently not used.
            data_fold_idx:      the index of the datafold to be loaded in the dataset.
        """
        
        self.subject_under_test = subject_under_test
        self.subj_idxed_feature_frames = load_feature_data(subject_under_test, detected_dir=data_dir, condition=condition_dic[subject_under_test], test=test, evaluating=evaluating, use_holdout=use_holdout, data_fold_idx=data_fold_idx)
        self.feature_start_index = 2        
        self.feature_size = 512 + 61 + 35 + 54 + 40 + 10
        self.feature_end_index = self.feature_start_index + self.feature_size
        self.threshold_percentile = threshold_percentile
        self.include_delta_feature = include_delta_feature
        
        self.targets_frames = {}
        for data_file_id in data_files:
            self.targets_frames[data_file_id] = pd.read_csv(data_files[data_file_id])
        self.target_idx = target_idx

        # Mappings to convert reward values to classes for training.
        if self.target_idx == 1:  
            self.weight_mappings =  {-5:0, -1:1, 6:2, 0:3}
            self.binary_weight_mappings1 = {-5:0, -1:0, 6:1, 0:2}
            self.binary_weight_mappings2 = {-5:0, -1:1, 6:1, 0:2}
        elif self.target_idx == 2: 
            self.weight_mappings =  {0:0, 1:1}
                
        self.frames_pre = frames_pre
        self.frames_after = frames_after
        self.window_size = self.frames_pre + self.frames_after + 1
        self.class_weights = list(np.ones(3))
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
        
        # Preprocess data into numpy arrays
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
                        annot_target =  np.array(curr_dataframe.iloc[data_idx, self.feature_end_index-10:self.feature_end_index]).astype('float')
                        event_mask = []
                        frame_num = int(curr_dataframe.iloc[idx, 0])
                        event_mask.append(frame_num//50)

                        mean_max_reaction_level = np.mean(np.amax(np.clip(img_feature[:,512+61:512+61+35],0.0,1.0), axis=1))

                        if prev_img_feature is None:
                            delta_feature = img_feature
                        else:
                            delta_feature = img_feature - prev_img_feature
                        prev_img_feature = img_feature

                        # Whether or not include delta feature with regards to previous frame.
                        # Currently not used.
                        if self.include_delta_feature:
                            self.processed_training_data.append(np.concatenate((img_feature, delta_feature), axis=0))
                        else:
                            self.processed_training_data.append(img_feature)

                        self.annot_targets.append(annot_target.flatten())
                        tar_idx = int(curr_dataframe.iloc[idx, 0])
                        self.targets_data.append(int(targets_frame.iloc[tar_idx, self.target_idx]))
                        self.reaction_levels.append(mean_max_reaction_level)
                        self.event_masks.append(np.array(event_mask))
                        self.subj_ids.append(subj_id)
                                    
        # Convert to np array for better indexing.
        self.processed_training_data = np.array(self.processed_training_data)
        self.targets_data = np.array(self.targets_data)
        self.reaction_levels = np.array(self.reaction_levels)
        self.event_masks = np.array(self.event_masks)
        self.subj_ids = np.array(self.subj_ids)
        self.annot_targets = np.array(self.annot_targets)

        # Filter classes, only keep the timesteps with non-zero rewards (events occurred).
        sampled_indices = np.squeeze(np.argwhere(abs(self.targets_data) > 0))

        self.processed_training_data = self.processed_training_data[sampled_indices]
        self.targets_data = self.targets_data[sampled_indices]
        self.reaction_levels = self.reaction_levels[sampled_indices]
        self.event_masks = self.event_masks[sampled_indices]
        self.subj_ids = self.subj_ids[sampled_indices]
        self.annot_targets = self.annot_targets[sampled_indices]

        # Originally used for sampling data in which humans have high reaction levels.
        # Currently not used, keeping all reaction data.
        # Get the threshold by percentile
        if self.threshold_percentile > 0:
            self.threshold = 0.5
            sampled_indices = np.squeeze(np.argwhere(self.reaction_levels > self.threshold))

            # Sample by thresholded indices
            self.processed_training_data = self.processed_training_data[sampled_indices]
            self.targets_data = self.targets_data[sampled_indices]
            self.reaction_levels = self.reaction_levels[sampled_indices]
            self.event_masks = self.event_masks[sampled_indices]
            self.subj_ids = self.subj_ids[sampled_indices]
            self.annot_targets = self.annot_targets[sampled_indices]

        self.data_size = int(self.processed_training_data.shape[0])


    # Balance the data instances in training by changing sample weights.
    def balance_data_by_class(self):
        labels = [self.weight_mappings[target] for target in self.targets_data]
        counts, _ = np.histogram(labels, bins=range(len(self.class_weights)+1))
        self.label_distribution = np.divide(counts,sum(counts))
        class_weights = np.divide(1.0/len(self.weight_mappings), np.divide(counts,sum(counts)))
        class_weights = np.divide(class_weights, sum(class_weights))
        self.update_sample_weights(class_weights)


    # Update the sample weights of classes based on their distribution.
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
        # Logic for loading one training instance.
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
            binary_labels1 = np.array(target) # just a placeholder, not used in regression
            binary_labels2 = np.array(target) # just a placeholder, not used in regression

        return (img_feature, label, subj_id, event_mask, binary_labels1, binary_labels2, annot_target)
        