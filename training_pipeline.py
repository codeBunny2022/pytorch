"""
1. Design the model (input, output size, forward pass)
2. Construct loss and optimizer
3. Training Loop
    - Forward pass: compute prediction
    - Backward pass: gradients
    - update weights
"""

import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


def forward(x):
    return w * x


def loss(y, y_predicted):
    return ((y - y_predicted) ** 2).mean()


learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iters):  # prediction =forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)
    # gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"epoch{epoch+1}:w ={w:.3f},loss={l:.8f}")
print(f"prediction after training f(10): {forward(5):.5f}")
