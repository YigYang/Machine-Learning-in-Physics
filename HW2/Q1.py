import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(6)

# Baseline ordinary MLP
# Symmetric model: h(x) = 1/2(g(x)+g(-x))

# Even function test
def f1(x):
    return torch.cos(3*x) + 0.3*x**2

def f2(x):
    return torch.abs(x) + 0.2*torch.cos(6*x)

# Data (n -> sample size, x_max -> maximum x, one_sided -> only sample one side or not)
def make_dataset(n=64, x_max=2.0, one_sided=False, noise_std=0.05, target_fn=f1):
    if one_sided:
        # train on x >= 0
        x = torch.rand(n, 1) * x_max
    else:
        x = (2 * torch.rand(n, 1) - 1) * x_max

    y = target_fn(x) + noise_std * torch.randn_like(x)
    return x, y

# Baseline MLP
class MLP(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x)

# symmetric MLP
class EvenMLP(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.g = MLP(width)

    def forward(self, x):
        return 0.5 * (self.g(x) + self.g(-x))

# Training and Evaluation
def train(model, x_train, y_train, epoch=1500, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epoch):
        opt.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()

    return model

def mse(model, x, y):
    with torch.no_grad():
        return torch.mean((model(x) - y) ** 2).item()

def run(target_fn=f1, one_sided=False, width=32):
    x_train, y_train = make_dataset(n=64, one_sided=one_sided, target_fn=target_fn)
    x_test = torch.linspace(-2, 2, 400).unsqueeze(1)
    y_test = target_fn(x_test)

    baseline = MLP(width=width)
    even_net = EvenMLP(width=width)

    train(baseline, x_train, y_train)
    train(even_net, x_train, y_train)

    baseline_mse = mse(baseline, x_test, y_test)
    even_mse = mse(even_net, x_test, y_test)

    print(f"Baseline test MSE: {baseline_mse:.6f}")
    print(f"EvenNet  test MSE: {even_mse:.6f}")

    with torch.no_grad():
        yb = baseline(x_test)
        ye = even_net(x_test)

    plt.figure(figsize=(7, 4))
    plt.scatter(x_train.numpy(), y_train.numpy(), s=18, alpha=0.7, label="train data")
    plt.plot(x_test.numpy(), y_test.numpy(), label="true f(x)")
    plt.plot(x_test.numpy(), yb.numpy(), label="baseline MLP")
    plt.plot(x_test.numpy(), ye.numpy(), label="even MLP")
    plt.legend()
    plt.title(f"one_sided={one_sided}, target={target_fn.__name__}")
    plt.show()

# Test Cases
print("Experiment 1: balanced training set")
run(target_fn=f1, one_sided=False)

print("Experiment 2: only train on x>=0")
run(target_fn=f1, one_sided=True)

print("Experiment 3: another even target")
run(target_fn=f2, one_sided=True)

print("Experiment 4: another even target")
run(target_fn=f2, one_sided=False)
