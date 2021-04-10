# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from data import Dataset
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup, Trainer, TrainingArguments
import numpy as np
import random

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', 
                                          bos_token='<|startoftext|>', 
                                          eos_token='<|endoftext|>', 
                                          pad_token='<|pad|>')

# # Test for longest sequence
# max_flavour = max([len(tokenizer.encode(card)) for card in cards])

# Instantiate Dataset
dataset = Dataset(data, tokenizer, max_length=max_flavour)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Loading the model configuration and setting it to the GPT2 standard settings.
config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

# Instantiate Model
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
model.resize_token_embeddings(len(tokenizer))

for parameter in model.parameters():
    parameter.requires_grad = False

for i, m in enumerate(model.transformer.h):        
    #Only un-freeze the last n transformer blocks
    if i >= 6:
        for parameter in m.parameters():
            parameter.requires_grad = True 

for parameter in model.transformer.ln_f.parameters():        
    parameter.requires_grad = True

for parameter in model.lm_head.parameters():        
    parameter.requires_grad = True

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()

# This step is optional but will enable reproducible runs.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We wil create a few variables to define the training parameters of the model
# epochs are the training rounds
# the warmup steps are steps at the start of training that are ignored
# every x steps we will sample the model to test the output

epochs = 4
warmup_steps = 1e2
sample_every = 100

# AdamW is a class from the huggingface library, it is the optimizer we will be using, and we will only be instantiating it with the default parameters. 
optimizer = AdamW(model.parameters(),
                  lr = 5e-4,
                  eps = 1e-8
                )

training_args = TrainingArguments(
        output_dir='./outputs',
        num_train_epochs=4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=100,
        weight_decay=.01,
        logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)