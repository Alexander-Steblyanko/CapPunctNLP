import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
import matplotlib.pyplot as plt
import transformers
import os
import pandas as pd
from settings import *


class Dataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.df = pd.read_hdf(filepath, "df")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return [torch.tensor(v) for v in self.df.loc[index].values]


class PunctuationModel(nn.Module):
    def __init__(self, pretrained_model_name, freeze_bert=False, lstm_dim=-1):
        super(PunctuationModel, self).__init__()

        # Bert Layer
        self.bert_layer = transformers.XLMRobertaModel.from_pretrained(pretrained_model_name)
        self.bert_layer.gradient_checkpointing_enable()
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Bidirectional LSTM
        if lstm_dim == -1:
            self.hidden_size = bert_dim
        else:
            self.hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=self.hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)

        # Linear Layer
        self.linear = nn.Linear(in_features=self.hidden_size * 2, out_features=len(punc_dict))

    def forward(self, x, x_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])
        #print(x.shape)
        #print(attn_masks.shape)
        out = self.bert_layer(x, attention_mask=x_masks)[0]
        #print(out.shape)
        #out, _ = torch.utils.checkpoint.checkpoint(self.lstm, out)
        out, _ = self.lstm(out)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    print("Start load!")
    ds = pd.read_hdf(dataset_path, "df")
    print("End load!")

    # FIND NONSTANDART LENS
    # item = ds.df['x'].map(len)
    # print(item)
    # item = item.loc[lambda x : x != 384]
    # print(item)
    # item = item.index
    # print(item)
    # item = ds.df.loc[item]
    # print(item)
    # print(ds.__len__())

    # FIND SPECIFIC LINE
    from pdata import reinterpret_tokens
    x, y, x_mask, y_mask = ds.loc[8]
    print(x)
    print(y)
    print(x_mask)
    print(y_mask)
    print(reinterpret_tokens(x, y, x_mask, y_mask))

    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(bert_model_name)
    print([tokenizer.decode(xi) for xi in x])

    # FIND WEIGHTS
    # amounts = []
    # for i in punc_dict.values():
    #     item = ds['y'].map(lambda x: x.count(i))
    #     count = item.sum()
    #     #print(count)
    #     amounts.append(count)
    # count = ds['y_mask'].map(lambda x: x.count(0)).sum()
    # amounts.append(count)
    # print(amounts)
    # tot = sum(amounts[:-1])
    # weights = [tot / (a * len(punc_dict)) for a in amounts]
    # print(weights)

    # FIND
