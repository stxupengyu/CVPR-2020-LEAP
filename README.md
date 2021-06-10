### Reproducibility:
python main.py --dataset AAPD
python main.py --dataset RCV
python main.py --dataset Eurlex

### configuration file
Please confirm the corresponding configuration file£º
config_AAPD
config_RCV
config_Eurlex

### configuration file describe
epochs: 10
lstm_hidden_dimension: 150
batch_size: 64
d_a: 100 #attention dim
emb_size: 300
GPU: True
GPU_Number: 2
load_path: '/data/xiaolin/dataset/RCV/'#data path
data_token: ''#no use in current code
lr: 0.001
scale: 120 # para of CosFace Loss
quantile: 0.8 #tail label ratio 
gamma: 0.1 #center learning rate for LEAP


