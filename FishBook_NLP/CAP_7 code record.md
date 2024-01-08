# CAP_7

---

## 1.code 1

```python
import sequence

(x_train, t_train), (x_test, t_test) = \
    sequence.load_data('addition.txt', seed=1984)
char2id, id2char = sequence.get_vocab()

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)

print(x_train[0])
print(t_train[0])

print([id2char[i] for i in x_train[0]])
print([id2char[i] for i in t_train[0]])
```

---

## 2.Seq2seq + reverse + peeky

```python
import torchnm

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from util import *
import sequence
import ptb

WORD_SIZE = 16
HIDDEN_SIZE = 128
BATCH_SIZE = 128
EPOCH = 50


class MyDataset(Dataset):
    def __init__(self, questions, answers):
        # questions:(N, L_q)
        # answers:(N, L_a)
        self.questions = torch.LongTensor(questions)
        self.answers = torch.LongTensor(answers)

        self.questions = self.questions.flip(dims=[1])  # reverse
        self.answers = self.answers.flip(dims=[1])  # reverse

        self.len = len(questions)

    def __getitem__(self, index):
        return self.questions[index], self.answers[index]
    
    def __len__(self):
        return self.len


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, wordvec_size, hidden_size) -> None:
        super(Encoder, self).__init__()
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.embed = torch.nn.Embedding(num_embeddings=V, embedding_dim=D)
        self.lstm = torch.nn.LSTM(input_size=D, hidden_size=H)
    
    def forward(self, xs):
        # (N, L)
        xs = self.embed(xs)  # (N, L, D)
        xs = xs.permute(1, 0, 2)
        
        outputs, (hn, cn) = self.lstm(xs)
        return hn  # bi*num_layers, N, H
    

class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, wordvec_size, hidden_size) -> None:
        super(Decoder, self).__init__()
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.embed = torch.nn.Embedding(num_embeddings=V, embedding_dim=D)
        self.lstm = torch.nn.LSTM(input_size=H+D, hidden_size=H)
        self.fc = torch.nn.Linear(in_features=H+H, out_features=V)
    
    def forward(self, ts, hn):
        # xs:(N, L), hn:(bi*num_layers, N, H)
        ts = self.embed(ts)  # (N, L, D)
        ts = ts.permute(1, 0, 2)  # (L, N, D)

        h_copy = hn
        h_copy = h_copy.repeat(ts.shape[0], 1, 1)  # (L, N, H)
        ts = torch.concat((h_copy, ts), dim=2)# peeky

        cn = torch.zeros_like(hn)
        outs, (h, c) = self.lstm(ts, (hn, cn))  # outs:(L, N, bi*H)
        outs = torch.concat((h_copy, outs), dim=2)  # peeky
        outs = outs.permute(1, 0, 2)# outs:(N, L, bi*H)

        outs = self.fc(outs)  # (N, L, V)
        return outs


class Seq2seq(torch.nn.Module):
    def __init__(self):
        super(Seq2seq, self).__init__()
        self.encoder = Encoder(vocab_size, WORD_SIZE, HIDDEN_SIZE)
        self.decoder = Decoder(vocab_size, WORD_SIZE, HIDDEN_SIZE)
    
    def forward(self, xs, decoder_xs):
        # xs: (N, L_q)
        # ts: (N, L_a)
        hn = self.encoder(xs)
        outs = self.decoder(decoder_xs, hn)
        return outs  # (N, L, V)


(x_train, t_train), (x_test, t_test) = \
    sequence.load_data('addition.txt', seed=1984)
char2id, id2char = sequence.get_vocab()
vocab_size = len(char2id)

train_dataset = MyDataset(x_train, t_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = MyDataset(x_test, t_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_data_len = len(test_dataset)

cpu = torch.device('cpu')
gpu = torch.device('cuda:0')

model = Seq2seq().to(gpu)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

correct_rate_list = []
for epoch in range(EPOCH):
    epoch_mean_loss = 0
    for idx, my_data in enumerate(train_loader, 0):
        xs, ts = my_data
        xs, ts = xs.to(gpu), ts.to(gpu)
        # xs: (N, L_q) ts: (N, L_a)
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]
        outs = model(xs, decoder_xs)  # (N, L, V)
        outs = outs.reshape(-1, vocab_size)
        decoder_ts = decoder_ts.flatten()
        loss = criterion(outs, decoder_ts)
        epoch_mean_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch={epoch}, epoch_mean_loss={epoch_mean_loss / len(train_loader)}")

    with torch.no_grad():
        correct_cnt = 0
        for idx, test__data in enumerate(test_loader, 0):
            test_xs, test_ts = test__data  # xs: (N, L_q) ts: (N, L_a)
            test_xs, test_ts = test_xs.to(gpu), test_ts.to(gpu)
            test_decoder_xs, test_decoder_ts = test_ts[:, :-1], test_ts[:, 1:]
            outs = model(test_xs, test_decoder_xs)  # (N, L, V)
            outs = outs.argmax(dim=2)  # (N, L)
            # test_decoder_ts: (N, L)
            for idx_batch, (test_x, out, test_decoder_t) in enumerate(zip(test_xs, outs, test_decoder_ts), 0):
                if (out.equal(test_decoder_t)):
                    correct_cnt += 1
                if (idx_batch < 5):
                    quest_str = ''
                    pred_str = ''
                    ans_str = ''
                    for i in test_x.tolist():
                        quest_str += id2char[i]
                    for j, z in zip(out.tolist(), test_decoder_t.tolist()):
                        pred_str += id2char[j]
                        ans_str += id2char[z]
                    print(quest_str)
                    print(pred_str)
                    print(ans_str)
                    print('----')
        correct_rate_list.append(correct_cnt / test_data_len)
        print(f'Epoch={epoch}, correct rate={correct_cnt / test_data_len}')

draw_pict(correct_rate_list, 0, EPOCH)


```



