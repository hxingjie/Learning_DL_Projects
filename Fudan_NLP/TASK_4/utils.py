import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def data_preprocess(file_path):
    # input:
    #'hello NNP B-NP TYPE'
    #'world VBZ B-VP TYPE'

    # output:
    # sentences: ['hello world', ...]
    # labels: ['TYPE TYPE', ...]
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sentences = []
    labels = []

    sentence = ''
    label = ''
    for line in lines:
        if line == '-DOCSTART- -X- -X- O\n':  # 忽略开头标识
            continue
        if line == '\n':  # 空行表示一个句子的结束
            if sentence != '':
                sentences.append(sentence.strip(' '))  # 去掉前后的空格
                labels.append(label.strip(' '))  # 去掉前后的空格
                sentence = ''
                label = ''
            continue
        # 'EU NNP B-NP B-ORG'
        # 'rejects VBZ B-VP O'
        line = line.strip('\n')
        words = line.split()
        sentence += words[0] + ' '
        label += words[3] + ' '
    
    return sentences, labels

def create_dict(sentences):
    word2id = {'_':0}
    id2word = {0:'_'}

    for sentence in sentences:
        sentence = sentence.strip(' ')
        words = sentence.split(' ')
        for word in words:
            if word not in word2id:
                idx = len(word2id)
                word2id[word] = idx
                id2word[idx] = word
    
    return word2id, id2word
        
def make_tensor(word2id, id2word, label2id, sentences, labels):
    corpus = []
    idxLabels = []
    seq_lengths = []
    max_length = -1
    for sentence in sentences:
        words = sentence.split(' ')
        tmp = []
        for word in words:
            if word == '':
                print()
            tmp.append(word2id[word])
        corpus.append(tmp)
        seq_len = len(tmp)
        seq_lengths.append(seq_len)
        max_length = max(max_length, seq_len)

    for label in labels:
        single_labels = label.split(' ')
        cur_labels = []
        for single_label in single_labels:
            if single_label == '':
                print()
            cur_labels.append(label2id[single_label])
        idxLabels.append(cur_labels)
    N = len(corpus)
    corpus_padding = torch.zeros(N, max_length, dtype=torch.long)
    idxLabels_padding = torch.zeros(N, max_length, dtype=torch.long)
    for i, (cs, ids, seq_len) in enumerate(zip(corpus, idxLabels, seq_lengths), 0):
        corpus_padding[i][:seq_len] = torch.LongTensor(cs)
        idxLabels_padding[i][:seq_len] = torch.LongTensor(ids)

    seq_lengths = torch.LongTensor(seq_lengths)

    seq_lengths, sorted_idx = seq_lengths.sort(dim=0, descending=True)
    corpus_padding = corpus_padding[sorted_idx]
    idxLabels_padding = idxLabels_padding[sorted_idx]
    
    return corpus_padding, idxLabels_padding, seq_lengths, max_length

def compute_f1(outs:list, labels:list, num_labels):
    # outs:list, (N, L_no_padding)
    # labels:list, (N, L_no_padding)
    val_TP, val_FP, val_FN = [], [], []
    for _ in range(num_labels):
        val_TP.append(0)
        val_FP.append(0)
        val_FN.append(0)
    for out, label in zip(outs, labels):
        for o, l in zip(out, label):
            # o:type_1, l:type_2
            if o == l:
                val_TP[o] += 1
            else:
                val_FP[o] += 1
                val_FN[l] += 1
    # val_TP: [type1_tp, type2_tp, ...]
    val_f1 = [0 for _ in range(num_labels)]
    for i in range(num_labels):
        if val_TP[i] == 0 and val_FP[i] == 0 and val_FN[i] == 0:
            val_f1[i] = 0
        else:
            val_f1[i] =  val_TP[i] / (val_TP[i] + (0.5 * ( val_FP[i] + val_FN[i])))
    
    return sum(val_f1) / len(val_f1)

def draw_pict(y_data):
    x = np.arange(0, len(y_data), 1, dtype=np.int32)
    y = np.array(y_data)

    plt.plot(x, y, linestyle='-')
    plt.show()

class MyDataSet(Dataset):
    def __init__(self, sentences, labels):
        self.sentences, self.labels = sentences, labels
        self.len = len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return self.len



