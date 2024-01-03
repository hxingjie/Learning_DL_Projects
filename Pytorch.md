## pytorch
```python
X = torch.Tensor()  # 创建Tensor，默认float类型的Tensor

X = torch.tensor()  # 创建Tensor，使用函数创建
函数原型为def tensor(data: Any, dtype: Optional[_dtype]=None, device: Device=None, requires_grad: _bool=False) -> Tensor: ...

X.require_grad = True  # 直接设置属性
X.require_grad_(True)  # 通过方法设置属性

X.grad  # 获取梯度，也是Tensor
X.data  # 获取值
X.item()  # 获取值

optimizer.zero_grad()
outputs = net(images.to(device))
loss = loss_function(outputs, labels.to(device))
loss.backward()
optimizer.step()

# 卷积、池化
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

# 二分类
sigmoid + BCEloss
softmax + CrossEntropyLoss, 注意，pytorch中的CrossEntropyLoss已经集成了softmax

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

torch.nn.MaxPool2d()
torch.nn.AvgPool2d()
torch.nn.AdaptiveMaxPool2d()
torch.nn.AdaptiveAvgPool2d()

self.fc1 = torch.nn.Linear(in_features=hidden_size, out_features=vocab_size)
self.fc1.weight.data = self.fc0.weight.data  # share weights
# torch.nn.Linear的weight为[out_features, hidden_size]

torch.concat((tensor1, tensor2, ...), dim=channel)
torch.flatten(tensor1)

x.shape
x.shape[0]
x.reshape(2,3)
x.size
x.ndim

h.unsqueeze(2)  # 在dim=2增加维度
h.repeat(1, 1+neg_sz, 1)  # 复制，参数是指对应维度复制多少次

torch.sum()
torch.mean()

tensor.permute(1, 0)
tensor.clone()
tensor.detach()
tensor.numpy()

torch.equal()函数接受两个张量作为输入，返回一个布尔值
torch.eq()函数接受两个张量作为输入，返回一个新的布尔张量

torch.randn((7, 3))  # 从标准正态分布中取数，size = (7, 3)
torch.matmul(c, W)  # 矩阵乘法

print(h.data)  # return tensor
print(h.item())  # return a number, h must only has one element

# to gpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # gpu
context, sample, label = context.to(device), sample.to(device), label.to(device)
outputs = model(context, sample)

loss = criterion(outputs, label)
optimizer.zero_grad()
loss.backward()
optimizer.step()

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
---
## 数据处理
```python
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

train_dataset = MyDataset(contexts, target, neg_samples)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

```
---
## 基本函数
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
## 绘图
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1, dtype=float)  # [0, 6), 步长为0.1
y1 = np.sin(x)
y2 = np.cos(x)

plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()  # 添加图例
plt.show()

plt.plot(x, y1, linestyle="-", label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
plt.show()
```
---
## 常用操作
```python
X = np.array([1, 2, 3, 4, 5, 6])
print(X[np.array([1, 3, 5])])  # 使用np.array访问指定的元素

print(X % 2 == 0)
# 使用dtype为bool类型的np.array访问指定的元素(取出True对应的元素)
# X % 2 == 0:[False  True False  True False  True]
print(X[X % 2 == 0])

```
---
## some bug
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

```

