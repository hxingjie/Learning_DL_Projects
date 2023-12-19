# CAP 4

---

**src.py**

```python
import torch

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from util import *
import ptb


WINDOW_SIZE = 5
HIDDEN_SIZE = 100
MAX_EPOCH = 150
BATCH_SIZE = 256
NEG_SZ = 7


class MyDataset(Dataset):
    def __init__(self, contexts, target, neg_samples):
        self.contexts = torch.LongTensor(contexts)  # (Seq, window_size * 2)
        
        target = torch.LongTensor(target)  # (Seq, )
        target = target.reshape(-1, 1)  # (Seq, 1)
        neg_samples = torch.LongTensor(neg_samples)  # (Seq, neg_sz)
        self.samples = torch.concat((target, neg_samples), dim=1)  # (Seq, 1+neg_sz)
        
        labels = [1] + [0 for _ in range(NEG_SZ)]
        labels = torch.Tensor(labels)  # (1+neg_sz*0)
        labels = labels.unsqueeze(dim=0)
        self.labels = labels.repeat(len(contexts), 1)  # (Seq, 1+neg_sz)
        
        self.len = len(contexts)

    def __getitem__(self, index):
        return self.contexts[index], self.samples[index], self.labels[index]

    def __len__(self):
        return self.len


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        super(CBOW, self).__init__()
        self.in_embed = torch.nn.Embedding(num_embeddings=V,
                                           embedding_dim=H)
        self.out_embed = torch.nn.Embedding(num_embeddings=V,
                                            embedding_dim=H)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, context, sample):
        # context: (B, window_size * 2)
        # sample: (B, 1+neg_sz)
        window_size = int(context.shape[1] / 2)
        neg_sz = sample.shape[1]-1

        in_embed = self.in_embed(context)  # context: (B, window_size * 2, hidden_size)
        h:torch.Tensor = torch.sum(in_embed, dim=1)  # h: (B, hidden_size)
        h /= window_size * 2  # h: (B, hidden_size)
        h = h.unsqueeze(1)
        h = h.repeat(1, 1+neg_sz, 1)  # h: (B, 1+neg_sz, hidden_size)
        
        out_embed = self.out_embed(sample)  # target: (B, 1+neg_sz, hidden_size)
        # 对应元素相乘
        outputs = h * out_embed  # outputs: (B, 1+neg_sz, hidden_size)
        outputs = torch.sum(outputs, dim=2)  # outputs: (B, 1+neg_sz)

        outputs = self.sigmoid(outputs)

        return outputs
    

# text = 'You say hello abd I say hi.'
# corpus, word2id, id2word = preprocess(text)
corpus, word2id, id2word = ptb.load_data('train')
corpus = corpus[:500000]
print("Load data is over.")

vocab_size = len(word2id)

contexts, target = create_contexts_target(corpus, window_size=WINDOW_SIZE)

neg_samples = get_neg_samples(corpus, target, id2word, NEG_SZ)  # np.array
print("Data preparation completed.")

train_dataset = MyDataset(contexts, target, neg_samples)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # gpu
model = CBOW(vocab_size, HIDDEN_SIZE).to(device)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

list_loss = []

for epoch in range(MAX_EPOCH):
    epoch_loss_sum = 0
    for idx, my_data in enumerate(train_loader, 0):
        context, sample, label = my_data
        context, sample, label = context.to(device), sample.to(device), label.to(device)
        outputs = model(context, sample)  # (Seq, 1+neg_sz) with sigmoid

        loss = criterion(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss_sum += loss.item()

    list_loss.append(epoch_loss_sum / BATCH_SIZE)
    print(f"Epoch={epoch}, Epoch mean loss={epoch_loss_sum / BATCH_SIZE}")

draw_pict(list_loss, 0, MAX_EPOCH)

model = model.to(torch.device('cpu'))
torch.save(model.in_embed.weight.clone().detach(), 'in_embed.pth')

```

**analogy.py**

```python
from util import *
import ptb

import torch

corpus, word2id, id2word = ptb.load_data('train')
word_vecs = torch.load('in_embed.pth')
word_vecs = word_vecs.numpy()

querys = ('you', 'year', 'car', 'toyota')
for query in querys:
    print(query + ':')
    most_similar(query, word2id, id2word, word_vecs, 5)
    print()

analogy('king', 'man', 'queen', word2id, id2word, word_vecs)
print()
analogy('take', 'took', 'go', word2id, id2word, word_vecs)
print()
analogy('car', 'cars', 'child', word2id, id2word, word_vecs)
print()
analogy('good', 'better', 'bad', word2id, id2word, word_vecs)
print()

```

