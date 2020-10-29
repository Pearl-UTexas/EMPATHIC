# Code for paper "The EMPATHIC Framework for Task Learning from Implicit Human Feedback"


## Requirements
All modules require Python 3.6 or above. 

- [install anaconda](https://docs.anaconda.com/anaconda/install/)
- [install pytorch](https://pytorch.org/get-started/locally/)
- [install openface2.0](https://github.com/TadasBaltrusaitis/OpenFace)


To install all Python dependencies, run:
```
python -m pip install --upgrade -r requirements.txt
```


---------------------------------------------------------------------------------------

## Robotaxi Environment: Robotaxi
(Game Logic Modified from repository: [snake-ai-reinforcement](https://github.com/YuriyGuts/snake-ai-reinforcement))


Test Robotaxi and control by yourself (using the arrow keys):
```
python play.py --agent human
```

Watch a psuedo optimal agent:
```
python play.py --agent val-itr
```

Watch an agent with mixed policy:
```
python play.py --agent mixed
```

## Data collection script

Script to play back recordings for collection human reactions (different condition number corresponds to different episode):
```
python start_data_collection.py --cond [0,1,..,5] 
```

## Playback Recordings
To playback recorded behavior (use --save_frames to save individual game frames as images):
```
python render_from_log.py --log_file <log_file_name> [--original] [--save_frames]
```

## Test Online Learning

Specify the path to your OpenFace installation in `start_openface.bash`

Run online learning (a webcam that can see your face is required):
```
python online_learning.py
```
(you may need to manually kill the process when it finishes)


---------------------------------------------------------------------------------------

## Supervised Learning
Since the dataset used in this paper is collected by ourselves and exceeds the maximum size limit of submission, the data is not included here and the training scripts below are not runnable for now. They will be available as soon as the final version of paper is released.

Important files:
- `network_facs.py`: network architecture
- `data_loader_facs.py`: data loading logic (preprocessing) for per-subject datasets
- `data_loader_all.py `: data loading logic (preprocessing) for single-model dataset
- `train_mlp_net_facs.py`: training script  for per-subject datasets
- `train_mlp_net_facsall.py`: training script for single-model dataset

## Train Supervised Learning for preidicting RL Statistics from Facial Features
```
$ python train_mlp_net_facs.py <subj_id>
```
or 
```
$ python train_mlp_net_facsall.py
```
