import os
import torch


# Punctuation dictionary
# Yeet the -?
punc_dict = {'_': 0, ',': 1, '.': 2, '?': 3, '-': 4,
        '_Up': 5, ',_Up': 6, '._Up': 7, '?_Up': 8, '-_Up': 9}

# Weights (calc in model.py)
weights = torch.tensor([0.12286, 2.183, 2.3084, 28.504225, 14.865,
                        1.356, 12.11708, 23.28645, 236.3064, 52.598])
#weights = torch.tensor([0.1, 2, 2, 5, 7, 1.3, 6, 11, 20, 13])


# Special Token IDs
spec_tokens = {
    'START_SEQ': 0,
    'PAD': 1,
    'END_SEQ': 2,
    'UNKNOWN': 3
}

freeze = True
batch_size = 48
bert_model_name = 'xlm-roberta-base'
learning_rate = 5e-6
max_rate = 5e-4
decay = 0
num_epochs = 3
num_workers = 4
#val_frequency = 100000

dataset_path = os.path.expanduser("data/TEDtalks.h5")
model_load_path = os.path.expanduser("model/load.path")
model_save_path = os.path.expanduser("model/load.path")

sequence_len = 384
bert_dim = sequence_len*2

#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)
