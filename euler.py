from cmath import cos, sin, e
from matplotlib import pyplot as plt

def edo_euler(f, x0, y0, b, n):
    x = x0
    y = y0
    h = (b - x0) / n

    xs, ys = [x0], [y0]
    for i in range(n):
        y = y + h * f(x, y)
        x = x + h
        xs.append(x)
        ys.append(y)

    return xs, ys

def edo_euler_melhorado(f, x0, y0, b, n):
    x = x0
    y = y0
    h = (b - x0) / n

    xs, ys = [x0], [y0]
    for i in range(n):
        y = y + h * (f(x, y) + f(x + h, y + h * f(x, y))) / 2
        x = x + h
        xs.append(x)
        ys.append(y)

    return xs, ys

def edo_runge_kutta(f, x0, y0, b, n):
    x = x0
    y = y0
    h = (b - x0) / n

    xs, ys = [x0], [y0]
    for i in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        x = x + h
        xs.append(x)
        ys.append(y)

    return xs, ys

if __name__ == '__main__':
    x0 = 0
    y0 = 2
    b = 2.5
    n = 100
    dydx = lambda x, y: y*(x**2) - (y**2)*x

    xs, ys = edo_euler(dydx, x0, y0, b, n)
    xs2, ys2 = edo_euler_melhorado(dydx, x0, y0, b, n)
    xs3, ys3 = edo_runge_kutta(dydx, x0, y0, b, n)

    f, ax = plt.subplots(1)
    ax.plot(xs, ys)
    ax.plot(xs2, ys2, linestyle='--')
    ax.plot(xs3, ys3, linestyle='-.')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Solução numérica da EDO y\' = yx² - y²x')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=x0)
    ax.grid()
    plt.show()