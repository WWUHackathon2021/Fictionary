#!/usr/bin/env python3

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

import torch
from transformers import (
    AdamW, 
    Trainer, 
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForPreTraining
)
import numpy as np
import time
import sys
import argparse

# Constants
SPECIAL_TOKENS  = { "bos_token": "<|BOS|>",
                    "eos_token": "<|EOS|>",
                    "unk_token": "<|UNK|>",                    
                    "pad_token": "<|PAD|>",
                    "sep_token": "<|SEP|>"}
MODEL           = 'gpt2' #{gpt2, gpt2-medium, gpt2-large, gpt2-xl}

MAXLEN          = 75

def parse_all_args():
    """
    Parses commandline arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("word",
                    help="Which word would you like defined?",
                    type=str
                )

    parser.add_argument("--train_on",
                    help="Generate definition or train model?",
                    type=bool,
                    default=False
                )
    return parser.parse_args()

def get_tokenizer(special_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL) #GPT2Tokenizer

    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def get_model(tokenizer, special_tokens=None, load_model_path=None):

    #GPT2LMHeadModel
    if special_tokens:
        config = AutoConfig.from_pretrained(MODEL, 
                                            bos_token_id=tokenizer.bos_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            sep_token_id=tokenizer.sep_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            output_hidden_states=False)
    else: 
        config = AutoConfig.from_pretrained(MODEL,                                     
                                            pad_token_id=tokenizer.eos_token_id,
                                            output_hidden_states=False)    

    #----------------------------------------------------------------#
    model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)

    if special_tokens:
        #Special tokens added, model needs to be resized accordingly
        model.resize_token_embeddings(len(tokenizer))

    if load_model_path:
        model.load_state_dict(torch.load(load_model_path))

    model.cuda()
    return model

def define(model, word, num_return=10):
    prompt = SPECIAL_TOKENS['bos_token'] + word + SPECIAL_TOKENS['sep_token']
    tokenizer = get_tokenizer(special_tokens=SPECIAL_TOKENS)
    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cuda")
    generated = generated.to(device)

    model.eval()

    # Top-p (nucleus) text generation (10 samples):
    sample_outputs = model.generate(generated, 
                                do_sample=True,   
                                min_length=50, 
                                max_length=MAXLEN,
                                top_k=30,                                 
                                top_p=0.7,        
                                temperature=0.9,
                                repetition_penalty=2.0,
                                num_return_sequences=num_return
                                )
    definitions = []
    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        a = len(word)
        print("{}: {}\n\n".format(word,  text[a:]))

        definitions.append(text[a:])
    
    return definitions

def get_model_for_api(weights_path='/home/pashbyl/Fictionary/outputs/pytorch_model.bin'):
    tokenizer = get_tokenizer(special_tokens=SPECIAL_TOKENS)
    model = get_model(
            tokenizer,
            special_tokens=SPECIAL_TOKENS,
            load_model_path=weights_path,
        )
    return model

def main(args):
    from data import Dataset

    args = parse_all_args()
    tokenizer = get_tokenizer(special_tokens=SPECIAL_TOKENS)
    
    if args.train_on:
        model = get_model(tokenizer, 
            special_tokens=SPECIAL_TOKENS,
        )

        # Instantiate Dataset
        train_dataset = Dataset('/home/dawc/Development/data/train.csv', tokenizer, MAXLEN)
        dev_dataset = Dataset('/home/dawc/Development/data/valid.csv', tokenizer, MAXLEN)

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

        # AdamW is a class from the huggingface library, it is the optimizer we will be using, and we will only be instantiating it with the default parameters. 
        optimizer = AdamW(model.parameters(),
                        lr = 5e-4,
                        eps = 1e-8
                        )

        training_args = TrainingArguments(
                output_dir='./outputs/dict1_epoch4',
                num_train_epochs=5,
                per_device_train_batch_size=25,
                per_device_eval_batch_size=25,
                gradient_accumulation_steps=5,
                warmup_steps=100,
                weight_decay=.01,
                logging_dir='./logs'
        )

    #---------------------------------------------------#
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            tokenizer=tokenizer
        )
    #---------------------------------------------------#

        trainer.train()
        trainer.save_model()
        
    else:
        model = get_model(tokenizer, 
                special_tokens=SPECIAL_TOKENS,
                load_model_path='/home/pashbyl/dict2_epoch1_small.bin'
                )

        now = time.time()
        define(model,args.word)
        print('Time to define was {0} seconds'.format(round(time.time()-now,3)))

if __name__ ==  "__main__":
    main(sys.argv)