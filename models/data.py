#!/usr/bin/env python3

from os import read
import pandas as pd
import torch
from torch.utils import data
import numpy as np
import csv

class Dataset(data.Dataset):
    def __init__(self, data, tokenizer,max_length, gpt2_type="gpt2"):
        defs,words = [],[]

        with open(data, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter='\t')
            for i,row in enumerate(reader):
                if i == 0: continue
                words.append(row[0])
                defs.append(row[1])
        
        self.tokenizer = tokenizer
        self.defs = defs
        self.words = words
        self.max_length = max_length
        
    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):

        input = "<|BOS|>" + self.words[idx] + "<|SEP|>" + self.defs[idx] + "<|EOS|>"
        encodings_dict = self.tokenizer(input,                                   
                                   truncation=True, 
                                   max_length=self.max_length, 
                                   padding="max_length")

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids), 
                'attention_mask': torch.tensor(attention_mask)}
