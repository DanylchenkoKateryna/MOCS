import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# Основна функція для n=7
def f(x, n):
    return n * np.sin(np.pi*n*x)


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
    x = np.linspace(-np.pi, np.pi, 1000)
    return sum(abs(f(xi, n) - fourier_series(xi, N, n)) / abs(f(xi, n)) for xi in x) / 1000


# Функція для обчислення абсолютної похибки
def absolute_error(N, n):
    x = np.linspace(-np.pi, np.pi, 1000)
    return sum(abs(f(xi, n) - fourier_series(xi, N, n)) for xi in x) / 1000

# Функція для побудови гармонік
def plot_harmonics(N,n):
    x = np.linspace(0, np.pi, 1000)

    plt.figure(figsize=(14, 10))
    a0 = (1 / (2 * np.pi)) * integrate.quad(lambda x: f(x, n), 0, np.pi)[0]
    harmonic = [a0 / 2 for _ in x]
    plt.plot(x, harmonic, label="Harmonic 0")

    for k in range(1, N + 1):
        ak = a_k(k, n)
        bk = b_k(k, n)
        harmonic = [ak * np.cos(k * val) + bk * np.sin(k * val) for val in x]
        plt.plot(x, harmonic, label=f"Harmonic {k}")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

# Функція для побудови графіка ряду Фур'є
def plot_fourier(N,n):
    x = np.linspace(0, np.pi, 1000)
    y_original = [f(val,n) for val in x]
    y_fourier = [fourier_series(xi, N, n)for xi in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_original, label="Original plot")
    plt.plot(x, y_fourier, label=f"Fourier Series of Order {N}", color="orange", lw=1)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

# Функція для побудови спектру коефіцієнтів Фур'є
def plot_spectrum(n):
    a_coefficients = []
    b_coefficients = []
    ks = list(range(11))

    for k in ks:
        if k == 0:
            ak_value = a_k(k, n)
            bk_value = 0
        else:
            ak_value = a_k(k, n)
            bk_value = b_k(k, n)
        a_coefficients.append(abs(ak_value))
        b_coefficients.append(abs(bk_value))

    # Побудова графіка спектру
    plt.figure(figsize=(10, 6))
    plt.stem(ks, a_coefficients, "b", markerfmt="bo", basefmt=" ", label="|a_N|")
    plt.stem(ks[1:], b_coefficients[1:], "r", markerfmt="ro", basefmt=" ", label="|b_N|")
    plt.title("Частотний спектр коефіцієнтів Фур'є")
    plt.xlabel("N")
    plt.ylabel("Амплітуда")
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
    abserror = absolute_error(N,n)
    relerror = relative_error(N,n)

    # Побудова графіка
    plot_harmonics(N,n)
    plot_fourier(N,n)
    plot_spectrum(n)

    # Збереження результатів у файл
    with open("fourier_results.txt", "w") as file:
        file.write(f"Order N: {N}\n")
        file.write(f"a0 = {a0}\n")
        for k in range(1, N + 1):
            file.write(f"a{k} = {a_k(k, n)}, b{k} = {b_k(k, n)}\n")
        file.write(f"Relative Error: {abserror}\n")
        file.write(f"Adsolute Error: {relerror}\n")


if __name__ == "__main__":
    main()
