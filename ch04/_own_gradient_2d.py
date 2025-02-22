import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4
    return f(x + h) - f(x - h) / (2 * h)


def function_2(x):
    # return x[0]**2 + x[1]**2
    return np.sum(x**2)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val

    return grad


# グラフ描画はうまく行ってない

# if __name__ == "__main__":
#     x0 = np.arange(-2, 2.5, 0.25)
#     x1 = np.arange(-2, 2.5, 0.25)
#     X, Y = np.meshgrid(x0, x1)

#     X = X.flatten()
#     Y = Y.flatten()

#     grad = numerical_gradient(function_2, np.array([X, Y]).T).T

#     plt.figure()
#     plt.quiver(X, Y, -grad[0], -grad[1], angles="xy", color="#666666")
#     plt.xlim([-2, 2])
#     plt.ylim([-2, 2])
#     plt.xlabel("x0")
#     plt.ylabel("x1")
#     plt.grid()
#     plt.draw()
#     plt.show()
