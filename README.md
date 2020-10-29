# Code for paper "[The EMPATHIC Framework for Task Learning from Implicit Human Feedback](https://arxiv.org/abs/2009.13649)"


## Clone the Repository

`git clone --recursive https://github.com/Pearl-UTexas/EMPATHIC.git`

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


## Test Online Learning

Specify the path to your OpenFace installation in `start_openface.bash`

Run online learning (a webcam that can see your face is required):
```
python online_learning.py
```
(you may need to manually kill the process when it finishes)


---------------------------------------------------------------------------------------

## Training Human Reaction Mappings
- Download the pre-processed [dataset]()

Important files:
- `network_facs.py`: network architecture
- `data_loader_facs.py`: data loading logic for per-subject datasets
- `data_loader_all.py `: data loading logic for single-model dataset (final evaluation)
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
