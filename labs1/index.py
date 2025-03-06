import numpy as np
import matplotlib.pyplot as plt

# Исходные данные
a = 1          # левая граница
b = 2          # правая граница
y_a = 1        # граничное условие y(1) = 1
bc_b = 15.806  # граничное условие y'(2) - y(2) = 15.806
n = 8         # число разбиений

# Шаг сетки
h = (b - a) / n

# Узлы сетки
x = np.linspace(a, b, n+1)

# Точное решение
def exact_solution(x):
    return x**4 + np.log(x)

# Правая часть уравнения
def f(x):
    return -16 * np.log(x)

# Формируем трехдиагональную матрицу системы
def create_matrix_and_rhs():
    # Размер матрицы (n+1) x (n+1)
    A = np.zeros((n+1, n+1))
    F = np.zeros(n+1)
    
    # Первое уравнение (граничное условие слева)
    A[0, 0] = 1
    F[0] = y_a
    
    # Внутренние узлы
    for i in range(1, n):
        x_i = x[i]
        
        # Коэффициенты для разностной схемы
        A[i, i-1] = x_i**2 / h**2 - x_i / (2*h)
        A[i, i] = -2 * x_i**2 / h**2 - 16
        A[i, i+1] = x_i**2 / h**2 + x_i / (2*h)
        
        F[i] = f(x_i)
    
    # Последнее уравнение (граничное условие справа)
    # y'(b) - y(b) = bc_b
    # Аппроксимируем y'(b) ≈ (y[n] - y[n-1])/h
    A[n, n-1] = -1/h
    A[n, n] = 1/h - 1
    F[n] = bc_b
    
    return A, F

# Решение системы уравнений
def solve_system():
    A, F = create_matrix_and_rhs()
    y_numerical = np.linalg.solve(A, F)
    return y_numerical

# Вычисление производной для численного решения
def calculate_derivative_at_b(y_numerical):
    return (y_numerical[-1] - y_numerical[-2]) / h

# Проверка граничных условий
def check_boundary_conditions(y_numerical):
    derivative_at_b = calculate_derivative_at_b(y_numerical)
    bc_b_calculated = derivative_at_b - y_numerical[-1]
    
    print(f"Граничное условие в x = {a}: y({a}) = {y_numerical[0]:.6f} (должно быть {y_a})")
    print(f"Граничное условие в x = {b}: y'({b}) - y({b}) = {bc_b_calculated:.6f} (должно быть {bc_b})")

# Вычисление погрешности
def calculate_error(y_numerical):
    y_exact = exact_solution(x)
    absolute_error = np.abs(y_numerical - y_exact)
    relative_error = absolute_error / (np.abs(y_exact) + 1e-10) * 100  # в процентах
    
    max_abs_error = np.max(absolute_error)
    max_rel_error = np.max(relative_error)
    
    std = (np.sum((y_numerical - y_exact)**2) / 7) ** 0.5
    
    print("\nПогрешности:")
    print(f"Максимальная абсолютная погрешность: {max_abs_error:.6f}")
    print(f"Максимальная относительная погрешность: {max_rel_error:.6f}%")
    print(f"Среднее отклонение: {std:.6f}")
    
    return absolute_error, relative_error

# Построение графика
def plot_results(y_numerical):
    # Генерируем больше точек для гладкого графика точного решения
    x_exact = np.linspace(a, b, 100)
    y_exact_smooth = exact_solution(x_exact)
    
    # Точное решение в узлах сетки
    y_exact_nodes = exact_solution(x)
    
    # График решений
    plt.plot(x_exact, y_exact_smooth, 'b-', label='Точное решение')
    plt.plot(x, y_numerical, 'ro--', label='Численное решение')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title(f'Сравнение численного и точного решений при n={n}')
    plt.legend()

    plt.savefig("./labs1/num_acc_1.jpg")

# Основная функция
def main():
    print("Решение краевой задачи методом сеток")
    print(f"Уравнение: x^2 y'' + xy' - 16y = -16 ln(x)")
    print(f"Граничные условия: y({a}) = {y_a}, y'({b}) - y({b}) = {bc_b}")
    print(f"Точное решение: y = x^4 + ln(x)")
    print(f"Число разбиений: {n}")
    
    # Получаем численное решение
    y_numerical = solve_system()
    
    # Выводим результаты
    print("\nЧисленное решение в узлах сетки:")
    for i in range(n+1):
        print(f"x[{i}] = {x[i]:.4f}, y[{i}] = {y_numerical[i]:.6f}, y_exact = {exact_solution(x[i]):.6f}")
    
    # Проверяем граничные условия
    check_boundary_conditions(y_numerical)
    
    # Строим графики
    plot_results(y_numerical)
    
    # выводим погрешность
    calculate_error(y_numerical)

if __name__ == "__main__":
    main()