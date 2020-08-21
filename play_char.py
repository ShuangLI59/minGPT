import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
import logging
import argparse
# make deterministic
from mingpt.utils import set_seed
set_seed(42)

from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig


import pdb
import os
import pickle

parser = argparse.ArgumentParser('play_char.py', description='Run minGPT experiments.')
parser.add_argument('--block_size', type=int, default=128, help='spatial extent of the model for its context')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--max_epochs', type=int, default=50, help='max epochs')
parser.add_argument('--input_data', type=str, default='data/input.txt', help='data')
parser.add_argument('--ckpt_path', type=str, default='checkpoint/input', help='data')
args = parser.parse_args()


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


def main(args):

    # set up logging
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
    )



    if 'comm' in args.input_data:
        if os.path.exists(args.input_data):
            print('loadig data from %s' % args.input_data)
            text = open( args.input_data, 'r').read()
        else:
            file = open(args.input_data,"w") 

            lang_data_dir = "/data/vision/torralba/ls-objectvideo/3iclr2021/communication/maddpg-implementations/multiagent-communication-pytorch/data/init-envs/data/num_objs_landmarks/envs_examples10000_land2_obj18_tarobj18-18_examples10000_lang_action.p"
            save_lang_data = pickle.load( open( lang_data_dir, "rb" ))
            for k,tem in enumerate(save_lang_data):
                if k%100==0:
                    print('gen data:', k, len(save_lang_data))
                length = np.max([len(tem['message0']), len(tem['message1'])])
                
                for i in range(length):
                    if tem['message0'][i] is not None:
                        file.write(tem['message0'][i] + '\n') 
                    if tem['message1'][i] is not None:
                        file.write(tem['message1'][i] + '\n') 
                file.write('\n')
            file.close()
            
            text = open( args.input_data, 'r').read()
    else:
        print('loadig data from %s' % args.input_data)
        # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
        text = open( args.input_data, 'r').read() # don't worry we won't run out of file handles
    
    train_dataset = CharDataset(text, args.block_size) # one line of poem is roughly 50 characters


    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)


    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(train_dataset)*args.block_size,
                          num_workers=4, ckpt_path=args.ckpt_path)
    trainer = Trainer(model, train_dataset, None, tconf, args)
    trainer.train()


if __name__ == '__main__':
    main(args)

