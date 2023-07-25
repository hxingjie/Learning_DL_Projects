import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True # 需要梯度

def func(x):
    # w是tensor，* 已经被重载了
    # * ： tensor operator * tensor
    # x会被自动类型转换为tensor
    # 
    return x * w

def loss(x, y):
    #return (func(x)-y) ** 2
    # x               y
    #   * -> y' -> {loss} -> loss
    # w
    return pow(func(x)-y,2)

if (__name__ == '__main__'):
    print('predict (before training)',4,func(4).item())

    for epoch in range(100):
        for x, y in zip(x_data, y_data):
            # loss_tensor = loss(x, y) # loss_tensor是一个tensor
            loss_tensor = pow(x*w-y,2)
            loss_tensor.backward() 


            print('\tgrad:', x, y, w.grad.item())
            w.data -= 0.01 * w.grad.data

            w.grad.data.zero_() # 权重里梯度的数据清零
        
        print('pregress:', epoch, loss_tensor.item())
    
    print('predict (after training)', 4, func(4).item())


