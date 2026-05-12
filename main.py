from draw_function import draw
import numpy as np

def f(t, x):
    return np.exp(-(t / (1 + x**2) + 0.001 * x))

def simpson(g, a, b, eps, max_iter):
    N = 2
    for i in range(max_iter):
        h = (b - a) / N
        x = np.linspace(a, b, N + 1)
        gs = g(x)
        S = gs[0] + gs[-1]
        S += 4 * sum(gs[1:-1:2])
        S += 2 * sum(gs[2:-1:2])
        S *= h / 3

        N2 = 2 * N
        h2 = (b - a) / N2
        x2 = np.linspace(a, b, N2 + 1)
        gs2 = g(x2)
        S2 = gs2[0] + gs2[-1]
        S2 += 4 * np.sum(gs2[1:-1:2])
        S2 += 2 * np.sum(gs2[2:-1:2])
        S2 *= h2 / 3

        if abs(S2 - S) < eps:
            return S2, N2
        N = N2

    return S2, N

def gauss_3_nodes(g, a, b, n=4):
    xi =  np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
    w = np.array([5/9, 8/9, 5/9])

    h = (b - a) / n
    S = 0.0
    for i in range(n):
        x_left = a + i * h
        x_mid = x_left + h / 2
        x_nodes = x_mid + (h / 2) * xi
        S += (h / 2) * np.sum(w * (g(x_nodes)))
    return S

def gauss_4_nodes(g, a, b, n=4):
    xi = np.array([-0.8611363, -0.3399810, 0.3399810, 0.8611363])
    w = np.array([0.3478548, 0.6521451, 0.6521451, 0.3478548])

    h = (b - a) / n
    S = 0
    for i in range(n):
        x_left = a + i * h
        x_mid = x_left + h / 2
        x_nodes = x_mid + (h / 2) * xi
        S += (h / 2) * np.sum(w * (g(x_nodes)))
    return S


if __name__ == "__main__":
    a = -1
    b = 0
    c = 0.5
    d = 1.5
    m = 20
    eps = 0.001
    tau = (d - c) / m
    ts = [c + j * tau for j in range(m + 1)]

    print("t        Simpson  N     Gauss3      Gauss4")
    for t in ts:
        g = lambda x: f(t, x)
        I_Simpson, N = simpson(g, a, b, eps, 20)
        I_Gauss3 = gauss_3_nodes(g, a, b)
        I_Gauss4 = gauss_4_nodes(g, a, b)

        print(f"{t:.2f}     {I_Simpson:.5f}  {N}    {I_Gauss3:.5f}     {I_Gauss4:.5f}")
        # print(f"Симпсон {I_Simpson:.5f} для t={t:.3f}")
        # print(f"Гаусс {I_Gauss:.5f} для t={t:.3f}\n")
        
