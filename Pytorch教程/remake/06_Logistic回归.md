```python
import torch
import torch.nn.functional as F

x_data = torch.tensor([[1.], [2.], [3.]])  # 输入是3行向量
y_data = torch.tensor([[0.], [0.], [1.]])


class LogisticModel(torch.nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()  # 构造函数的固定写法 (class_name, self)
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))  # Linear类写了__call__函数，所以可以这样调用
        return y_pred


model = LogisticModel()

criterion = torch.nn.BCELoss(size_average=False)  # 求损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    # 得出y_pred
    # 得出loss
    # backward
    # update
    y_pred = model(x_data)  # 调用call -> forward
    loss = criterion(y_pred, y_data)  # 标量
    print(epoch, loss.item())

    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 求梯度
    optimizer.step()  # 更新

print(model(torch.tensor([[4.]])))
print(model(torch.tensor([[4.]])).data)
print(model(torch.tensor([[4.]])).item())

```
