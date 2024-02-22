import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import json

WORD_EMB = 500
HIDDEN_SZ = 128
BATCH_SZ = 256
EPOCH = 50
OUT_FEATURES = 4  # {"neutral":0, "contradiction":1, "entailment":2, "-":3}
SEQ_SZ = 30

CPU = torch.device('cpu')
GPU = torch.device('cuda:0')


def text_preprocess(sent1, sent2, sent3, sent4):
    # sent: (N,)
    word2id = {'_':0}
    id2word = {0:'_'}

    corpus1 = []
    for sen in sent1:
        sen = sen.lower()
        sen = sen.strip(' ')
        words = sen.split(' ')
        cs = []
        for word in words:
            if len(word) == 0:
                continue
            if word[-1] == ',' or word[-1] == '.':
                word = word[:-1]
            if word not in word2id:
                idx = len(word2id)
                word2id[word] = idx
                id2word[idx] = word
            cs.append(word2id[word])
        corpus1.append(cs)

    corpus2 = []
    for sen in sent2:
        sen = sen.lower()
        sen = sen.strip(' ')
        words = sen.split(' ')
        cs = []
        for word in words:
            if len(word) == 0:
                continue
            if word[-1] == ',' or word[-1] == '.':
                word = word[:-1]
            if word not in word2id:
                idx = len(word2id)
                word2id[word] = idx
                id2word[idx] = word
            cs.append(word2id[word])
        corpus2.append(cs)
    
    corpus3 = []
    for sen in sent3:
        sen = sen.lower()
        sen = sen.strip(' ')
        words = sen.split(' ')
        cs = []
        for word in words:
            if len(word) == 0:
                continue
            if word[-1] == ',' or word[-1] == '.':
                word = word[:-1]
            if word not in word2id:
                idx = len(word2id)
                word2id[word] = idx
                id2word[idx] = word
            cs.append(word2id[word])
        corpus3.append(cs)

    corpus4 = []
    for sen in sent4:
        sen = sen.lower()
        sen = sen.strip(' ')
        words = sen.split(' ')
        cs = []
        for word in words:
            if len(word) == 0:
                continue
            if word[-1] == ',' or word[-1] == '.':
                word = word[:-1]
            if word not in word2id:
                idx = len(word2id)
                word2id[word] = idx
                id2word[idx] = word
            cs.append(word2id[word])
        corpus4.append(cs)
    
    return word2id, id2word, corpus1, corpus2, corpus3, corpus4


class MyDataset(Dataset):
    def __init__(self, inputs):
        labels, corpus1, corpus2 = inputs[0], inputs[1], inputs[2]
        for idx in range(len(corpus1)):
            if len(corpus1[idx]) > SEQ_SZ:
                corpus1[idx] = corpus1[idx][:SEQ_SZ]
            else:
                corpus1[idx].extend([0 for _ in range(SEQ_SZ-len(corpus1[idx]))])
        for idx in range(len(corpus2)):
            if len(corpus2[idx]) > SEQ_SZ:
                corpus2[idx] = corpus2[idx][:SEQ_SZ]
            else:
                corpus2[idx].extend([0 for _ in range(SEQ_SZ-len(corpus2[idx]))])

        self.labels, self.corpus1, self.corpus2 = torch.LongTensor(labels), torch.LongTensor(corpus1), torch.LongTensor(corpus2)
        self.len = len(self.labels)
    
    def __getitem__(self, index):
        return self.corpus1[index], self.corpus2[index], self.labels[index]

    def __len__(self):
            return self.len


class EISM(torch.nn.Module):
    def __init__(self, vocab_sz, hidden_sz, word_emb, out_features):
        super(EISM, self).__init__()
        self.emb = torch.nn.Embedding(num_embeddings=vocab_sz, embedding_dim=word_emb)
        self.lstm = torch.nn.LSTM(input_size=word_emb, hidden_size=hidden_sz, bidirectional=True)
        self.tanh = torch.nn.Tanh()
        self.lstm_out = torch.nn.LSTM(input_size=8*hidden_sz, hidden_size=2*hidden_sz)
        self.fc = torch.nn.Linear(in_features=8*hidden_sz, out_features=out_features)

    def forward(self, sent1, sent2):
        # sent1, sent2: (N, L)
        N = sent1.shape[0]
        L1, L2 = sent1.shape[1], sent2.shape[1]

        # --------input encoding--------
        sent1_emb = self.emb(sent1)  # (N, L, word_emb)
        sent2_emb = self.emb(sent2)
        sent1_emb = sent1_emb.permute(1, 0, 2)  # (L, N, word_emb)
        sent2_emb = sent2_emb.permute(1, 0, 2)

        # out: (L, N, 2*H), h: (2*num_l, N, H), c==h
        sent1_lstm, _ = self.lstm(sent1_emb)
        sent2_lstm, _ = self.lstm(sent2_emb)

        sent1_lstm = sent1_lstm.permute(1, 0, 2)  # (N, L1, 2*H)
        sent2_lstm = sent2_lstm.permute(1, 0, 2)  # (N, L2, 2*H)
        # --------input encoding--------

        # --------local inference--------
        e_sent = torch.matmul(sent1_lstm, sent2_lstm.permute(0, 2, 1))  # e_sent: (N, L1=i, L2=j)
        
        # tmp1: (N, L1, L2), tmp2: (N, L2, L1)
        a_sent = torch.matmul(torch.exp(e_sent) / torch.exp(e_sent).sum(dim=2).reshape(N, L1, 1).repeat(1, 1, L2), sent2_lstm)
        b_sent = torch.matmul(torch.exp(e_sent) / torch.exp(e_sent).sum(dim=1).reshape(N, 1, L2).repeat(1, L1, 1).permute(0, 2, 1), sent1_lstm)
        
        # a_sent: (N, L1, 2*H), b_sent: (N, L2, 2*H)
        m_a = torch.concat([sent1_lstm, a_sent, sent1_lstm-a_sent, sent1_lstm*a_sent], dim=2)
        m_b = torch.concat([sent2_lstm, b_sent, sent2_lstm-b_sent, sent2_lstm*b_sent], dim=2)
        # m_a: (N, L1, 8*H), m_b: (N, L2, 8*H)
        # --------local inference--------

        # --------inference composition--------
        out_a, _ = self.lstm_out(m_a)  # (N, L1, 2*H)
        out_a1 = torch.nn.functional.avg_pool1d(out_a.permute(0, 2, 1), out_a.shape[1]).reshape(N, -1)  # (N, 2*H, 1)->(N, 2*H)
        out_a2 = torch.nn.functional.max_pool1d(out_a.permute(0, 2, 1), out_a.shape[1]).reshape(N, -1)  # (N, 2*H, 1)->(N, 2*H)
        out_a = torch.concat([out_a1, out_a2], dim=1)  # (N, 4*H)
        
        out_b, _ = self.lstm_out(m_b)  # (N, L2, 2*H)
        out_b1 = torch.nn.functional.avg_pool1d(out_b.permute(0, 2, 1), out_b.shape[1]).reshape(N, -1)  # (N, 2*H, 1)->(N, 2*H)
        out_b2 = torch.nn.functional.max_pool1d(out_b.permute(0, 2, 1), out_b.shape[1]).reshape(N, -1)  # (N, 2*H, 1)->(N, 2*H)
        out_b = torch.concat([out_b1, out_b2], dim=1)  # (N, 4*H)

        out = torch.concat([out_a, out_b], dim=1)  # (N, 8*H)
        out = self.fc(out)  # (N, out_features)
        # --------inference composition--------

        return out


def draw_pict(y_data):
    x = np.arange(0, len(y_data), 1, dtype=np.int32)
    y = np.array(y_data)
    plt.plot(x, y, linestyle='-')
    plt.show()


def trainModel(loader, model, criterion, optimizer):
    loss_list = []
    for epoch in range(EPOCH):
        loss_sum = 0
        for idx, (sents1, sents2, labels) in enumerate(loader, 0):
            sents1, sents2, labels = sents1.to(GPU), sents2.to(GPU), labels.to(GPU)
            outs = model(sents1, sents2)
            loss = criterion(outs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
        mean_loss = loss_sum / len(loader)  # this epoch's every sample's mean loss
        loss_list.append(mean_loss)
        print(f'Epoch={epoch}, epoch mean loss={mean_loss}')
    draw_pict(loss_list)


def testModel(test_set, loader, model):
    with torch.no_grad():
        accu_sum = 0
        for idx, (sents1, sents2, labels) in enumerate(loader, 0):
            sents1, sents2, labels = sents1.to(GPU), sents2.to(GPU), labels.to(GPU)
            outs = model(sents1, sents2)
            # outs: (N, out_features), labels: (N,)
            outs = torch.argmax(outs, dim=1)
            accu_sum += torch.eq(outs, labels).sum().item()
        accu_rate = accu_sum / len(test_set)  # this epoch's accu sample cnt
        print(f'Accu rate={accu_rate}')

# tmp_train.jsonl tmp_test.jsonl
# snli_train.jsonl snli_test.jsonl
with open('./TASK_3/snli_train.jsonl', 'r') as f:
    train_json_data = [json.loads(line.strip('\n')) for line in f]
with open('./TASK_3/snli_test.jsonl', 'r') as f:
    test_json_data = [json.loads(line.strip('\n')) for line in f]
train_inputs = [[_["gold_label"], _["sentence1"], _["sentence2"]] for _ in train_json_data]
test_inputs = [[_["gold_label"], _["sentence1"], _["sentence2"]] for _ in test_json_data]

tmp_dict = {"neutral":0, "contradiction":1, "entailment":2, "-":3}
train_labels = [tmp_dict[_[0]] for _ in train_inputs]
test_labels = [tmp_dict[_[0]] for _ in test_inputs]

train_sent1 = [_[1] for _ in train_inputs]
train_sent2 = [_[2] for _ in train_inputs]
test_sent1 = [_[1] for _ in test_inputs]
test_sent2 = [_[2] for _ in test_inputs]

word2id, id2word, train_corpus1, train_corpus2, test_corpus1, test_corpus2 = text_preprocess(train_sent1, train_sent2, test_sent1, test_sent2)

train_inputs = (train_labels, train_corpus1, train_corpus2)
test_inputs = (test_labels, test_corpus1, test_corpus2)

train_dataset = MyDataset(train_inputs)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SZ, shuffle=True)

test_dataset = MyDataset(test_inputs)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SZ, shuffle=False)

vocab_sz = len(word2id)

model = EISM(vocab_sz, HIDDEN_SZ, WORD_EMB, OUT_FEATURES).to(GPU)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(),lr=0.0004, betas=(0.9, 0.999))

trainModel(train_loader, model, criterion, optimizer)
testModel(test_dataset, test_loader, model)


