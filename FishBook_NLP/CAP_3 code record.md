# CAP_3 code record

---

## 1.code 1

```python
c = torch.Tensor([[1, 0, 0, 0, 0, 0, 0]])
W = torch.randn((7, 3))
h = torch.matmul(c, W)
print(h)
```

---

## 2.code 2

```python
c = torch.Tensor([[1, 0, 0, 0, 0, 0, 0]])
layer = torch.nn.Linear(in_features=7, out_features=3, bias=False)
h:torch.Tensor = layer(c)
print(h.da)
print(h.item())
```

---

## 3.code 3

```python
c0 = torch.Tensor([[1, 0, 0, 0, 0, 0, 0]])
c1 = torch.Tensor([[0, 0, 1, 0, 0, 0, 0]])
in_layer = torch.nn.Linear(in_features=7, out_features=3, bias=False)
out_layer = torch.nn.Linear(in_features=3, out_features=7, bias=False)
h0:torch.Tensor = in_layer(c0)
h1:torch.Tensor = in_layer(c1)
h = 0.5 * (h0 + h1)
s = out_layer(h)

print(s)
```

---

## 4. code 4: CBOW

```python
import torch

import numpy as np

from util import *

WINDOW_SIZE = 1
HIDDEN_SIZE = 4
MAX_EPOCH = 1000

text = 'You say hello abd I say hi.'
corpus, word2id, id2word = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size=WINDOW_SIZE)

vocab_size = len(word2id)


class SimpleCBOW(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        super(SimpleCBOW, self).__init__()
        self.in_layer = torch.nn.Linear(V, H, bias=False)
        self.out_layer = torch.nn.Linear(H, V, bias=False)

    def forward(self, contexts):
        # contexts: (Seq, window_size, hidden_size)
        input_0 = contexts[:, 0, :]  # (Seq, hidden_size)
        input_1 = contexts[:, 1, :]  # (Seq, hidden_size)
        h0 = self.in_layer(input_0)
        h1 = self.in_layer(input_1)
        h = (h0 + h1) * 0.5
        outputs = self.out_layer(h)
        return outputs


model = SimpleCBOW(vocab_size, HIDDEN_SIZE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

target = convert_one_hot(target, vocab_size)  # (Seq, hidden_size)
contexts = convert_one_hot(contexts, vocab_size)  # (Seq, window_size, hidden_size)

list_loss = []

for epoch in range(MAX_EPOCH):
    contexts = torch.Tensor(contexts)  # (Seq, window_size, hidden_size)
    target = torch.Tensor(target)
    outputs = model(contexts)  # (Seq, hidden_size)
    loss = criterion(outputs, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    list_loss.append(loss.item())
    print(loss.item())

draw_pict(list_loss, 0, MAX_EPOCH)

word_vecs = model.in_layer.weight.permute(1, 0).clone()
word_vecs = word_vecs.detach()
word_vecs = word_vecs.numpy()
for word_id, word in id2word.items():
    print(word, word_vecs[word_id])
```

## 5.code5: Skip_Gram

```python
class SimpleSkip_Gram(torch.nn.Module):
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        super(SimpleSkip_Gram, self).__init__()
        self.in_layer = torch.nn.Linear(V, H, bias=False)
        self.out_layer = torch.nn.Linear(H, V, bias=False)

    def forward(self, target):
        # target: (Seq, vocab_size)
        h = self.in_layer(target)  # h: (Seq, hidden_size)
        outputs = self.out_layer(h)  # h: (Seq, hidden_size)
        return outputs


model = SimpleSkip_Gram(vocab_size, HIDDEN_SIZE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

target = convert_one_hot(target, vocab_size)  # (Seq, hidden_size)
contexts = convert_one_hot(contexts, vocab_size)  # (Seq, window_size, hidden_size)

list_loss = []

for epoch in range(MAX_EPOCH):
    contexts = torch.Tensor(contexts)  # (Seq, window_size, hidden_size)
    target = torch.Tensor(target)
    outputs = model(target)  # (Seq, hidden_size)
    loss = 0
    n = contexts.shape[1]
    for idx in range(contexts.shape[1]):
        context = contexts[:, idx, :]
        loss += criterion(outputs, contexts[:, idx, :])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    list_loss.append(loss.item())
    print(loss.item())

draw_pict(list_loss, 0, MAX_EPOCH)

word_vecs = model.in_layer.weight.permute(1, 0).clone()
word_vecs = word_vecs.detach()
word_vecs = word_vecs.numpy()
for word_id, word in id2word.items():
    print(word, word_vecs[word_id])
```

