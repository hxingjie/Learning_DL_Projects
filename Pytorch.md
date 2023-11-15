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

torch.eq(tensor1, tensor2) -> tensor_bool
torch.equal(tensor1, tensor2) -> bool

torch.nn.Linear()
torch.nn.Conv()

torch.nn.MaxPool2d()
torch.nn.AvgPool2d()
torch.nn.AdaptiveMaxPool2d()
torch.nn.AdaptiveAvgPool2d()

torch.concat((tensor1, tensor2, ...), dim=channel)
torch.flatten(tensor1)

```
---
## 基本函数
```python
x.shape
x.shape[0]
x.reshape(2,3)
x.size

x = np.array([1, 2, 3, 4])
np.arange(-5.0, 5.0, 0.1) -> array
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
```
---
## 绘图
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1, dtype=float)  # [0, 6), 步长为0.1
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, linestyle="-", label="sin")
plt.plot(x, y2, linestyle="--", label="cos")
```
---
## 常用操作
```python
np.random.rand(3, 2)  # 生成一个3行2列的数组，元素是[0, 1)之间的随机数
np.random.randn(0, 1000)  # 生成一个3行2列的数组，元素是标准正态分布(均值为0，标准差为1)的随机数
np.random.normal(0, 1, size=(3, 2))  # 生成一个3行2列的数组，元素是符合均值为0，标准差为1正态分布的随机数

X = np.array([1, 2, 3, 4, 5, 6])
print(X[np.array([1, 3, 5])])  # 使用np.array访问指定的元素

print(X % 2 == 0)
# 使用dtype为bool类型的np.array访问指定的元素(取出True对应的元素)
# X % 2 == 0:[False  True False  True False  True]
print(X[X % 2 == 0])

plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()  # 添加图例
plt.show()
```