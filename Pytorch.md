## 1.pytorch

### train & test
```python
# train
outpust = model(inputs)
loss = criterion(outputs, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()

# test
model.eval()  # 关闭dropout
with torch.no_grad():  # 不进行梯度计算
    outpust = model(inputs)
    loss = criterion(outputs, labels)
```

### 激活函数的选择
回归：
MSEloss. reduction='mean' or 'sum or 'none'
MSEloss(outs, labels). outs:(N, *), labels:(N, *)

回归用作分类: 
sigmoid + BCEloss. reduction='mean' or 'sum or 'none'
BCEloss(sigmoid(outs), labels). outs:(N, C), labels:(N, C)

多分类: 
softmax + CrossEntropyLoss(pytorch中的CrossEntropyLoss已经集成了softmax). reduction='mean' or 'sum or 'none'
CrossEntropyLoss(outs, labels). outs:(N, C), labels:(N, )

### 常用层
```python
torch.nn.Linear()
torch.nn.Conv()

torch.nn.LSTM()
# xs: L, N, H_{in}
# h_0: D * num_layers, N, H_{out}
# c_0: D * num_layers, N, H_{cell}

# outs: L, N, D * H_{out}, D * H_{out}:(forward, reverse)
# h_n: D * num_layers, N, H_{out}, D * num_layers: (layer1's forward, layer1's reverse, layer2's forward, layer2's reverse)
# c_n: D * num_layers, N, H_{cell}
h_0 = torch.zeros(self.num_layers * (2 if self.bi else 1), N, self.H).to(gpu)
c_0 = torch.zeros_like(h_0).to(gpu)
outs, (hn, cn) = self.lstm(xs, (h_0, c_0))
print(outs[-1, -1, :HIDDEN_SIZE].eq(hn[-2, -1]))
print(outs[0, -1, HIDDEN_SIZE:].eq(hn[-1, -1]))

seq_lens, sorted_idx = seq_lens.sort(dim=0, descending=True)
x, y = x[sorted_idx], y[sorted_idx]

# pack && pad
lstm_inputs = torch.nn.utils.rnn.pack_padded_sequence(input=embed_out, lengths=seq_lengths)  
h0 = torch.zeros(self.D * self.num_layers, N, self.H)
c0 = torch.zeros_like(h0)
lstm_out, (hn, cn) = self.lstm(lstm_inputs, (h0, c0))  # L, N, H
lstm_out, lengths_unpacked = torch.nn.utils.rnn.pad_packed_sequence(sequence=lstm_out, total_length=max_len)

torch.nn.MaxPool2d()
torch.nn.AvgPool2d()
torch.nn.AdaptiveMaxPool2d()
torch.nn.AdaptiveAvgPool2d()

torch.nn.Dropout: 每个数按概率置0
torch.nn.Dropout1d: 以最后一维为单位置0
torch.nn.Dropout2d: 以最后两维为单位置0
torch.nn.Dropout3d: 以最后三维为单位置0

# 卷积、池化的维度计算
# N = (W - F + (P_row + P_col)) / S + 1
# 卷积 padding = size // 2
# 最大池化 padding = (size - 1) // 2

self.features = nn.Sequential(
    nn.ZeroPad2d((1, 2, 1, 2)),  # 左补一列, 右补两列, 上补一行, 下补两行
    nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)
    # padding=2, 上下左右全部补2行(列)0
    # padding=(1,2), 上下各补一行0, 左右各补两列0
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2)
)
```

### share weights
```python
self.fc1 = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)
self.fc1.weight.data = self.fc0.weight.data  # share weights
# torch.nn.Linear的weight为[out_features, hidden_size]
```

### translate to gpu
```python
CPU = torch.device('cpu')
GPU = torch.device('cuda:0')
tensor.to(device)
model.to(device)
```

### save and load
```python
# save and load tensor
model = model.to(torch.device('cpu'))
torch.save(model.in_embed.weight.clone().detach(), 'in_embed.pth')
word_vecs = torch.load('in_embed.pth')

# save and load model
# 1
torch.save(model, 'lstm_model.pth')
model = torch.load('lstm_model.pth')

# 2
torch.save(model.state_dict(), 'lstm_state_dict.pth')

state_dict = torch.load('lstm_state_dict.pth')
model = RNNlm(vocab_size, HIDDEN_SIZE)
model.load_state_dict(state_dict)
```

### glove
```python
from torchtext.vocab import Vectors

# pretrain vectors in vectors.py
vectors = Vectors(name='glove.6B.100d.txt', cache='./glove.6B/')
words = [word for word in word2id.keys()]
vectors = vectors.get_vecs_by_tokens(words, lower_case_backup=True)
self.embed = torch.nn.Embedding.from_pretrained(vectors)
```

---
## 2.pytorch 常用函数
```python
X = torch.Tensor()  # 创建Tensor，默认float类型的Tensor
X = torch.LongTensor()  # 创建Tensor，LongTensor类型的Tensor

X = torch.tensor()  # 创建Tensor，使用函数创建
函数原型为def tensor(data: Any, dtype: Optional[_dtype]=None, device: Device=None, requires_grad: _bool=False) -> Tensor: ...

X.require_grad = True  # 直接设置属性
X.require_grad_(True)  # 通过方法设置属性

torch.concat((tensor1, tensor2, ...), dim=channel)
torch.flatten(tensor1)

x.shape
x.shape[0]
x.size
x.ndim

h.unsqueeze(2)  # 在dim=2增加维度
x.reshape(2,3)
h.repeat(1, 1+neg_sz, 1)  # 复制，参数是指对应维度复制多少次

torch.sum(dim=2)
torch.mean(dim=2)

tensor.permute(1, 0, 2)
tensor.clone()
tensor.detach()
tensor.numpy()

torch.equal()函数接受两个张量作为输入，返回一个布尔值
torch.eq()函数接受两个张量作为输入，返回一个新的布尔张量

torch.randn((7, 3))  # 从标准正态分布中取数，size = (7, 3)
torch.matmul(c, W)  # 矩阵乘法

Tensor.data  # return tensor
Tensor.item()  # return a number, h must only has one element
Tensor.grad  # 获取梯度

with torch.no_grad()
    ...
    ...
```
---
## 3.数据处理
```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, questions, answers):
        # questions:(N, L_q)
        # answers:(N, L_a)
        self.questions = questions
        self.answers = (answers
        self.len = len(questions)

    def __getitem__(self, index):
        return self.questions[index], self.answers[index]
    
    def __len__(self):
        return self.len

train_dataset = MyDataset(x_train, t_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_data_len = len(train_dataset)
```
---
## 4.numpy
```python
x.shape
x.shape[0]
x.reshape(2,3)
x.size

x = np.array([1, 2, 3, 4])
np.arange(-5.0, 5.0, 0.1) -> numpy.array
np.maximum(0, x) -> array
np.exp(-x) -> array
print(x, type(x), np.ndim(x), x.shape, x.shape[0])

# boardcast, basic operation will apply boardcast
x + y, x - y, x * y, x / y

np.max(x) -> float
np.sum(x) -> float

np.dot(x, y) -> array

x = x.flatten()  # 转换成一维数组
np.argmax(y)  # return maximum value's index

np.c_[x, y]  # 对应元素组合为列表，再组合为高一维的列表

np.random.rand(3, 2)  # 生成一个3行2列的数组，元素是[0, 1)之间的随机数
np.random.randn(0, 1000)  # 生成一个3行2列的数组，元素是标准正态分布(均值为0，标准差为1)的随机数
np.random.normal(0, 1, size=(3, 2))  # 生成一个3行2列的数组，元素是符合均值为0，标准差为1正态分布的随机数
```
---
## 5.绘图
```python
import numpy as np
import matplotlib.pyplot as plt

def draw_pict(y_data):
    x = np.arange(0, len(y_data), 1, dtype=np.int32)
    y = np.array(y_data)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("x & y")
    plt.legend()  # 添加图例
    plt.plot(x, y, linestyle='-')

    plt.show()

```
---
## 6.常用操作
```python
X = np.array([1, 2, 3, 4, 5, 6])
print(X[np.array([1, 3, 5])])  # 使用np.array访问指定的元素

print(X % 2 == 0)
# 使用dtype为bool类型的np.array访问指定的元素(取出True对应的元素)
# X % 2 == 0: [False  True False  True False  True]
print(X[X % 2 == 0])

```
---
## 7.some bug
```python
(x_train, t_train), (x_test, t_test) = \
    sequence.load_data('addition.txt', seed=1984)
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]  # reverse
x_train, x_test = x_train.copy(), x_test.copy()  # copy
'''
当您尝试从具有负步长的 NumPy 数组创建张量时, 
PyTorch 中通常会出现此错误。 
PyTorch 张量不支持负步幅，
因此解决方法是在将数组转换为 PyTorch 张量之前使用 array.copy() 创建数组的副本。
'''
---
## other tools
tensorboard
wandb

```

