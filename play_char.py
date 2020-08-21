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
from mingpt.utils import sample

parser = argparse.ArgumentParser('play_char.py', description='Run minGPT experiments.')
parser.add_argument('--block_size', type=int, default=128, help='spatial extent of the model for its context')
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


    # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
    text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
    train_dataset = CharDataset(text, args.block_size) # one line of poem is roughly 50 characters


    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=8, n_head=8, n_embd=512)
    model = GPT(mconf)


    # initialize a trainer instance and kick off training
    tconf = TrainerConfig(max_epochs=200, batch_size=512, learning_rate=6e-4,
                          lr_decay=True, warmup_tokens=512*20, final_tokens=200*len(train_dataset)*args.block_size,
                          num_workers=4)
    trainer = Trainer(model, train_dataset, None, tconf)
    trainer.train()


    # alright, let's sample some character-level shakespear
    context = "O God, O God!"
    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
    y = sample(model, x, 2000, temperature=0.9, sample=True, top_k=5)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    print(completion)



if __name__ == '__main__':
    main(args)

