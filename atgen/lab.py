import torch

class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.sigmoid(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return torch.sigmoid(x) * (1 - torch.sigmoid(x)) * grad_output
    
class sigmoid(torch.nn.Module):
    def forward(self, x):
        return Sigmoid.apply(x)
    

x = torch.randn(20, 1)
y = torch.randn(20, 1)

model = torch.nn.Sequential(
    torch.nn.Linear(1, 100),
    sigmoid(),
    # torch.nn.Sigmoid(),
    # torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    sigmoid(),
    # torch.nn.Sigmoid(),
    # torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    sigmoid(),
    # torch.nn.Sigmoid(),
    # torch.nn.ReLU(),
    torch.nn.Linear(100, 1)
)

optimizer = torch.optim.AdamW(model.parameters(), 0.001)
criterion = torch.nn.MSELoss()

for _ in range(10000):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    print(f'Loss: {loss.item()}')
