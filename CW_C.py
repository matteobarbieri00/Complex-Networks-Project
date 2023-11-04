import numpy as np
from random import *
import matplotlib.pyplot as plt

N = 5


def build_adj_n_cycle(n):
    matrix = []
    for i in range(n):
        matrix.append([])
    for i in range(n):
        for j in range(n):
            if j == i - 1 or j == i + 1:
                matrix[i].append(0.5)
            elif i == 0 and j == n - 1:
                matrix[i].append(0.5)
            elif i == n - 1 and j == 0:
                matrix[i].append(0.5)
            else:
                matrix[i].append(0)
    return np.matrix(matrix)


def initialize_state(i, n):
    init_state = []
    for j in range(n):
        init_state.append([])
    for j in range(n):
        if j == i:
            init_state[j].append(1)
        else:
            init_state[j].append(0)
    return np.matrix(init_state)


def evolution(c, s, T):
    s_t = s
    for t in range(T):
        s_t = c * s_t
    return s_t


def variation_distance(a, b):
    a_0 = a.transpose()
    # b_0 = b.transpose()
    a_1 = a_0.tolist()
    # b_1 = b_0.tolist()
    a_2 = []
    # b_2 = []
    for i in a_1[0]:
        a_2.append(i)
    # for i in b_1[0]:
    #    b_2.append(i)
    ab = []
    for i in range(len(a_2)):
        ab.append(abs(a_2[i] - b[i]))
    return sum(ab)


def calc_mixing_time(b, c, s, T, epsilon):
    s_t = s
    loops = 0
    for t in range(T):
        s_t = c * s_t
        if variation_distance(s_t, b) <= epsilon:
            return t
        else:
            loops += 1
            if loops == T - 1:
                return -1
            else:
                pass


tot_T = 30
eps = 0.1 / N
A = build_adj_n_cycle(N)
s_0 = initialize_state(1, N)
final_state = evolution(A, s_0, tot_T)
pi = []
for i in range(N):
    pi.append(1 / N)

print(calc_mixing_time(pi, A, s_0, tot_T, eps))


x = [i for i in range(N)]
y_0 = final_state.transpose()
y_1 = y_0.tolist()
y = []
for i in y_1[0]:
    y.append(i)

plt.plot(x, y)
plt.ylim(0, 1)
plt.xlabel("Vertexes")
plt.ylabel("Probability")
plt.show()
