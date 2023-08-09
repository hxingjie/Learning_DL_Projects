x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def func(x):
    return x * w

def cost(x_data, y_data):
    cost = 0
    for x, y in zip(x_data, y_data):
        cost += (func(x)-y) ** 2
    return cost / len(x_data)

# 1/N * sum((xn * w - yn)^2,n(1,N)) 对 w 求导
# 等于 1/N * sum( 2*xn ( xn * w - yn), n(1,N) )
def gradient(x_data, y_data):
    grad = 0
    for x, y in zip(x_data, y_data):
        grad += 2 * x * (x * w - y)
    return grad / len(x_data)

print('Predict (before training): ',4,func(4))

for i in range(100):
    # cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)# grad_val是loss对w的导数
    w -= 0.01 * grad_val# 0.01是学习率，w(new) = w(old) - 学习率*loss对w的导数
    print('Epoch:', i, 'w =', w)

print('Predict (after training):', 4, func(4))
