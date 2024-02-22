import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy
import math

from utils import *
from model import *

BATCH_SIZE = 32
FILE_PATH = './poetryFromTang.txt'
EPOCH = 200
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512

MAX_LEN = 6

CPU = torch.device('cpu')
GPU = torch.device('cuda:0')
USE_GPU = True

def train(epoch, myloader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    for (corpus, seq_len) in train_loader:
        # N, 7
        N = corpus.shape[0]
        if USE_GPU:
            corpus = corpus.to(GPU)
        x, y = corpus[:, :-1], corpus[:, 1:]
        seq_len, sorted_idx = seq_len.sort(dim=0, descending=True)
        x, y = x[sorted_idx], y[sorted_idx]
        outs = model(x, seq_len, MAX_LEN)
        outs = outs.reshape(N * outs.shape[1], outs.shape[2])
        y = y.flatten()
        loss = criterion(outs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    mean_loss = total_loss / len(myloader)
    print(f'Epoch={epoch}, mean loss = {mean_loss}')
    print(f'Epoch={epoch}, Perplexity = {math.exp(mean_loss)}')
    return total_loss / len(myloader)

if __name__ == '__main__':
    sentences = read_file(FILE_PATH)
    word2id, id2word = create_dict(sentences)
    corpus, seq_len = create_corpus(sentences, word2id)
    corpus, seq_len = make_tensor(corpus, seq_len)

    train_dataset = MyDataset(corpus, seq_len)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_data_len = len(train_dataset)
    
    model = Model(len(word2id), EMBEDDING_DIM, HIDDEN_SIZE)
    if USE_GPU:
        model = model.to(GPU)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999))

    for epoch in range(EPOCH):
        train(epoch, train_loader, model, criterion, optimizer)
        
    model.predict(word2id, id2word, '春', 28)
       