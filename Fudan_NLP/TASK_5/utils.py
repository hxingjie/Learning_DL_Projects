import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    sentences = []
    for line in lines:
        if line == '\n':
            continue
        line = line.strip('\n')
        sentences.append(line)
    return sentences

def create_dict(sentences):
    word2id = {' ':0}
    id2word = {0:' '}
    for sentence in sentences:
        # '巴山上峡重复重，阳台碧峭十二峰。荆王猎时逢暮雨，'
        for word in sentence:
            if (word == '，' or word == '。'):
                continue
            if word not in word2id:
                idx = len(word2id)
                word2id[word] = idx
                id2word[idx] = word
    return word2id, id2word

def create_corpus(sentences, word2id):
    corpus = []  # N, L
    seq_len = []  # N
    for sentence in sentences:
        cor = []
        for word in sentence:
            if word == '，' or word == '。':  # 该句结束
                tmp = len(cor)
                seq_len.append(tmp if tmp < 7 else 6)
                if tmp < 7:  # 该句不是七言
                    cor.extend([0 for _ in range(7 - tmp)])
                if tmp > 7:
                    cor = cor[:7]
                corpus.append(cor)
                cor = []
            else:
                cor.append(word2id[word])
    return corpus, seq_len

def make_tensor(corpus, seq_len):
    corpus = torch.LongTensor(corpus)
    seq_len = torch.LongTensor(seq_len)
    return corpus, seq_len

class MyDataset(Dataset):
    def __init__(self, corpus, seq_len):
        self.corpus = corpus  # N, L
        self.seq_len = seq_len  # N
        self.len = len(corpus)

    def __getitem__(self, index):
        return self.corpus[index], self.seq_len[index]
    
    def __len__(self):
        return self.len


