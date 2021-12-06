import torch
import numpy as np

def gradient_descent(alpha, iteration, f, x):
    x = torch.tensor(x, requires_grad=True)
    f_val = []
    ite = []
    for i in range(iteration):
        ite.append(i)
        val = f(x)
        f_val.append(val.item())
        val.backward()
        with torch.no_grad():
            x -= x.grad * alpha
            x.grad.zero_()
    ite.append(iteration)
    f_val.append(f(x).item())
    return ite, f_val


def newton_descent(x1, x2, alpha, iteration, f, f_d, f_dd):
    point_sequence_x1 = [x1]
    point_sequence_x2 = [x2]
    value_sequence = [f(x1, x2)]
    dd_inv = np.linalg.inv(f_dd(2,2))
    for i in range(iteration):
        d = f_d(x1, x2)
        newton_direction = 0 - np.matmul(dd_inv,d)
        newton_direction = newton_direction.tolist()
        x1 += alpha * newton_direction[0][0]
        x2 += alpha * newton_direction[0][1]
        point_sequence_x1.append(x1)
        point_sequence_x2.append(x2)
        value_sequence.append(f(x1, x2))
    return x1, x2, point_sequence_x1, point_sequence_x2, value_sequence
