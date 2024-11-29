import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# Основна функція для n=7
def f(x, n):
    return n * np.sin(np.pi * n * x)


# Функція для обчислення коефіцієнтів a_k
def a_k(k, n):
    result, _ = integrate.quad(lambda x: f(x, n) * np.cos(k * x), 0, np.pi)
    return (1 / np.pi) * result


# Функція для обчислення коефіцієнтів b_k
def b_k(k, n):
    result, _ = integrate.quad(lambda x: f(x, n) * np.sin(k * x), 0, np.pi)
    return (1 / np.pi) * result


# Функція для обчислення наближення функції рядом Фур'є до порядку N
def fourier_series(x, N, n):
    a0 = (1 / (2 * np.pi)) * integrate.quad(lambda x: f(x, n), 0, np.pi)[0]
    sum_f = a0
    for k in range(1, N + 1):
        ak = a_k(k, n)
        bk = b_k(k, n)
        sum_f += ak * np.cos(k * x) + bk * np.sin(k * x)
    return sum_f


def relative_error(N, n):
    x = np.linspace(0, np.pi, 1000)
    return np.sum(abs(f(x, n) - fourier_series(x, N, n)) / abs(f(x, n))) / 1000


# Функція для обчислення абсолютної похибки
def absolute_error(N, n):
    x = np.linspace(0, np.pi, 1000)
    return np.sum(abs(f(x, n) - fourier_series(x, N, n))) / 1000


# Функція для побудови графіка ряду Фур'є
def plot_fourier(N, n):
    x = np.linspace(0, np.pi, 1000)
    y_original = f(x, n)  # Original function values
    y_fourier = fourier_series(x, N, n)  # Fourier series values

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_original, label="Original Function", color='blue')
    plt.plot(x, y_fourier, label=f"Fourier Series of Order {N}", color="orange", lw=2)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Fourier Series Approximation")
    plt.grid(True)
    plt.legend()
    plt.show()


# Головна функція
def main():
    n = 7  # Номер студента
    N = 10  # Порядок ряду Фур'є
    print(f"Обчислюємо наближення для n = {n} та N = {N}")

    # Обчислення та виведення коефіцієнтів
    a0 = (1 / (2 * np.pi)) * integrate.quad(lambda x: f(x, n), 0, np.pi)[0]
    print(f"a0 = {a0}")
    for k in range(1, N + 1):
        ak = a_k(k, n)
        bk = b_k(k, n)
        print(f"a{k} = {ak}, b{k} = {bk}")

    # Оцінка похибки
    abserror = absolute_error(N, n)
    relerror = relative_error(N, n)
    print(f"Absolute Error: {abserror}")
    print(f"Relative Error: {relerror}")

    # Побудова графіка
    plot_fourier(N, n)

    # Збереження результатів у файл
    with open("fourier_results.txt", "w") as file:
        file.write(f"Order N: {N}\n")
        file.write(f"a0 = {a0}\n")
        for k in range(1, N + 1):
            file.write(f"a{k} = {a_k(k, n)}, b{k} = {b_k(k, n)}\n")
        file.write(f"Relative Error: {relerror}\n")
        file.write(f"Absolute Error: {abserror}\n")


if __name__ == "__main__":
    main()
