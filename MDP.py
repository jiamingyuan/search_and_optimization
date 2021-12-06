import numpy as np

def value_function(gamma, pi, num_states, transitions, rewards):
    square = np.zeros([num_states, num_states])
    id_mtx = np.identity(num_states)

    for s in range(num_states):
        square[s, :] = id_mtx[s, :] - gamma * transitions[int(pi[s])]

    inv = np.linalg.inv(square)

    V_s = np.dot(inv, rewards)
    return V_s


# Policy improvement
def policy_improvement(gamma, s, pi, num_states, num_actions, transitions, rewards):
    vals = np.repeat(-np.inf, num_actions)

    for a in range(num_actions):
        vals[a] = np.sum(transitions[a][:, s] * value_function(gamma, pi, num_states, transitions, rewards))

    return np.argmax(vals)


# Policy iteration
def policy_iteration(gamma, num_states, num_actions, transitions, rewards):
    pi = np.random.randint(num_actions, size=num_states)
    V = value_function(pi)

    while True:
        pi_new = np.repeat(1, num_states)

        for i in range(num_states):
            pi_new[i] = policy_improvement(gamma, i, pi, num_states, num_actions, transitions, rewards)
        V_new = value_function(gamma, pi_new, num_states, transitions, rewards)

        if all(V == V_new):
            break
        pi = pi_new
        V = V_new
    return V, pi


def value_iteration(gamma, num_states, num_actions, transitions, rewards):
    V = np.repeat(0, num_states)

    while True:
        vals = np.full((num_actions, num_states), -np.inf)
        for i in range(num_actions):
            vals[i, :] = [np.sum(transitions[i][:, s] * V) for s in range(num_states)]
        V_new = np.array([rewards[s] + gamma * np.amax(vals[:, s]) for s in range(num_states)])
        if all(V_new == V):
            vals = np.full([num_actions, num_states], -np.inf)
            for i in range(num_actions):
                vals[i, :] = [np.sum(transitions[i][:, s] * V_new) for s in range(num_states)]
            pi_opt = np.array([np.argmax(vals[:, s]) for s in range(num_states)])
            break
        V = V_new

    return V, pi_opt