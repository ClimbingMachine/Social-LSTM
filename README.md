# Social-LSTM

## Project Motivation
Reference: https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf. 
This paper introduced one of the most innovative pedestrian trajectory prediction algorithm (Social-LSTM). 
However, it seems that there are multiple implementations of Social-LSTM,
and different metrics (ADE/FDE) reported by the different papers using the same datasets:
  
  1. Original Implementation: https://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf with ADE for ETH dataset (0.50);
  2. Social-GAN Implementation (co-authored by #1): https://arxiv.org/pdf/1803.10892.pdf with ADE for ETH dataset (0.73); and
  3. Social-STGCNN: https://arxiv.org/pdf/1803.10892.pdf with ADE for ETH dataset (1.09).

## Train and Test

Train a model using the ETH dataset inside the datasets folder, this involves running main.py

```
python3 main.py
```

Feel free to change the default parameters from lines 96-106 in main.py. 

## Current Best Results
So far, we just test the ETH datasets (leave-one-out approach).

To see the best results for 10 epochs, please refer to the log file: 

```
./saved_models/training_log_20_20_grid_size_enc_size_64.txt
```

## To-Do
Try other network parameters and grid sizes to fune-tune the Social-LSTM
