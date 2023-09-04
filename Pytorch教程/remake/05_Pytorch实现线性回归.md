```python
import torch

x_data = torch.tensor([[1.], [2.], [3.]])  # 输入是3行向量
y_data = torch.tensor([[2.], [4.], [6.]])


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()  # 构造函数的固定写法 (class_name, self)
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)  # Linear类写了__call__函数，所以可以这样调用
        return y_pred


model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)  # 求损失
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    # 得出y_pred
    # 得出loss
    # backward
    # update
    y_pred = model(x_data)  # 调用call -> forward
    loss = criterion(y_pred, y_data)  # 标量
    print(epoch, loss)

    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 求梯度
    optimizer.step()  # 更新

print(model.linear.weight.item())
print(model.linear.bias.item())

print(model(torch.tensor([[4.]])))
print(model(torch.tensor([[4.]])).data)
print(model(torch.tensor([[4.]])).item())

```
