import torch

x=torch.randn(3,requires_grad=True)
print(f"the value of x is {x}")

y=x+2
print(f"the value of y is {y}")

z=y*y*2
print(z)
z=z.mean()
print(f"the value of z is {z}")

z.backward() # dz/dx
print(f"the gradient values of gradient are {x.grad}")