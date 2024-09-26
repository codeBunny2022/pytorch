import numpy as np
import torch

# f = w * x
# f = 2 * x

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # weights


# model prediction
def forward(x):
    return w * x


# loss MSE
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()  # it returns the mean squared error


def gradient(x, y, y_predicted):
    return np.dot(2 * x, y_predicted - y).mean()


print(f"Prediction before training: f(10) ={forward(10):.3f}")

learning_rate = 0.02
n_iters = 10

for epoch in range(n_iters):
    y_pred = forward(X)
    l = loss(Y, y_pred)
    l.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad

    # zero gradients
    w.grad.zero_()

    if epoch % 1 == 0:
        print(f"epoch {epoch+1}:w={w:.3f},loss={l:.8f}")

print(f"Prediction after training : f(10)={forward(10):.3f}")
