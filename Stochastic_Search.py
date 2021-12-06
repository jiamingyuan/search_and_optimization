import torch
import numpy as np

def simulated_annealing(T, iteration, f, x, seed):
    torch.manual_seed(seed)
    x = torch.tensor(x)
    l = len(x)
    f_val = [f(x)]
    ite = [0]
    for k in range(1, iteration + 1):
        val = f(x)
        delta = torch.normal(0, 1, size=([l]))
        x_new = x+delta
        val_new = f(x_new)
        if val > val_new or torch.rand(1) < torch.exp((val - val_new) / (T / k)).item():
            x = x_new
        ite.append(k)
        f_val.append(f(x))
    return ite, f_val

def cross_entropy(k, iteration, f, x, seed):
    torch.manual_seed(seed)
    x = torch.tensor(x)
    l = len(x)
    f_val = [f(x).item()]
    ite = []
    mean = x
    cov = torch.diag(torch.tensor([1] * l))
    for i in range(iteration):
        ite.append(i)
        sample_x = []
        sample_val = []
        for j in range(k):
            sample = torch.tensor(np.random.multivariate_normal(mean, cov))
            val = f(sample)
            sample_x.append(list(sample))
            sample_val.append(val.item())
        samples = [(val, x) for x, val in zip(sample_x, sample_val)]
        samples.sort()
        elite_samples = [torch.tensor(x) for val, x in samples[: int(0.2 * k)]]
        elite_samples = torch.stack(elite_samples)
        mean = torch.mean(elite_samples, 0)
        cov = torch.cov(elite_samples.T)
        f_val.append(sum([val for val, x in samples[: int(0.2 * k)]])/int(0.2 * k))
    ite.append(iteration)
    return ite, f_val

def search_gradient(k, eta, iteration, f, x, seed):
    torch.manual_seed(seed)
    x = torch.tensor(x)
    l = len(x)
    f_val = [f(x)]
    ite = []
    mean = x
    mean.requires_grad = True
    cov = torch.diag(torch.tensor([1] * l))

    for i in range(iteration):
        ite.append(i)
        sum = torch.zeros(l)
        for j in range(k):
            sample = torch.normal(x, 1)
            sample.requires_grad = True
            val = f(sample)
            log_val = torch.log(val)
            log_val.backward()
            sum += val * sample.grad
            sample.requires_grad = False
        x -= eta * (sum.data / k)
        f_val.append(f(x))
    ite.append(iteration)
    return ite, f_val