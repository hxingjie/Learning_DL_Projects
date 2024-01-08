# CAP 5

---

## 1.RNNlm

```python
import torch

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from util import *
import ptb


HIDDEN_SIZE = 100
MAX_EPOCH = 1
BATCH_SIZE = 32
SEQ_SIZE = 50


class MyDataset(Dataset):
    def __init__(self, inputs, target):
        # inputs: (N) -> (-1, SEQ_SIZE)
        # target: (N) -> (-1, SEQ_SIZE)
        self.x, self.y = [], []
        beg = 0
        while (beg+SEQ_SIZE <= len(inputs)):
            self.x.append(inputs[beg:beg+SEQ_SIZE])
            self.y.append(target[beg:beg+SEQ_SIZE])
            beg += SEQ_SIZE
        self.x, self.y = torch.LongTensor(self.x), torch.LongTensor(self.y)
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


class RNNlm(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RNNlm, self).__init__()
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size)

        self.rnn = torch.nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=1)

        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, inputs):
        # inputs: torch.LongTensor, shape:(B, Seq)
        in_embed = self.embed(inputs)  # (B, Seq, H)
        in_embed = in_embed.permute(1, 0, 2)  # (Seq, B, H)
        outputs, _ = self.rnn(in_embed)  # outputs: (Seq, B, H)
        outputs = self.fc(outputs)  # outputs: (Seq, B, V)
        outputs = outputs.reshape(-1, self.vocab_size)  # outputs: (Seq*B, V)
        return outputs
    

# text = 'You say hello abd I say hi.'
# corpus, word2id, id2word = preprocess(text)
corpus, word2id, id2word = ptb.load_data('train')
corpus = corpus[:10001]
print("Load data is over.")

vocab_size = len(word2id)
# [2, 1, 4, 2, 8, 10]
# [2, 1, 4, 2, 8    ]
# [   1, 4, 2, 8, 10]
inputs, target = corpus[:-1], corpus[1:]

train_dataset = MyDataset(inputs, target)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # gpu
model = RNNlm(vocab_size, HIDDEN_SIZE).to(device)
criterion = torch.nn.CrossEntropyLoss(reduction='mean')  # softmax + CEloss
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

list_loss = []

for epoch in range(MAX_EPOCH):
    epoch_loss_sum = 0
    for idx, my_data in enumerate(train_loader, 0):
        x, y = my_data
        # x: (B, Seq)
        # y: (B, Seq)
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        y = y.flatten()
        # outputs: (Seq*B, V), y: (Seq*B)
        loss = criterion(outputs, y)  # mini_batch's mean loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss_sum += loss.item()
    list_loss.append(np.exp(epoch_loss_sum / len(train_loader)))
    print(f"Epoch={epoch}, Epoch ppl={np.exp(epoch_loss_sum / len(train_loader))}")

draw_pict(list_loss, 0, MAX_EPOCH)





```



