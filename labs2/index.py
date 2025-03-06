import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

# Исходные данные
a = 1          # левая граница
b = 2          # правая граница
y_a = 1        # граничное условие y(1) = 1
bc_b = 15.806  # граничное условие y'(2) - y(2) = 15.806
n = 8          # число узлов коллокации

# Точное решение
def exact_solution(x):
    return x**4 + np.log(x)

# Правая часть уравнения
def f(x):
    return -16 * np.log(x)

# Узлы коллокации (используем равномерную сетку)
collocation_points = np.linspace(a, b, n)

# Базисные функции (полиномиальный базис)
def basis_function(k, x):
    """
    Полиномиальная базисная функция k-ой степени
    """
    return x**k

# Производные базисных функций
def basis_function_derivative(k, x, order=1):
    """
    Производная полиномиальной базисной функции k-ой степени порядка order
    """
    if order == 1:
        return k * x**(k-1) if k > 0 else 0
    elif order == 2:
        return k * (k-1) * x**(k-2) if k > 1 else 0
    else:
        return 0

# Формирование системы уравнений метода коллокаций
def collocation_system(coeffs):
    # Формируем систему уравнений
    equations = np.zeros(n+1)
    
    # Учитываем левое граничное условие y(a) = y_a
    y_at_a = sum(coeff * basis_function(k, a) for k, coeff in enumerate(coeffs))
    equations[0] = y_at_a - y_a
    
    # Для каждой точки коллокации (кроме граничных) составляем уравнение
    for i, x_i in enumerate(collocation_points[1:-1], 1):
        # Вычисляем значение y, y' и y'' в точке коллокации
        y = sum(coeff * basis_function(k, x_i) for k, coeff in enumerate(coeffs))
        y_prime = sum(coeff * basis_function_derivative(k, x_i, 1) for k, coeff in enumerate(coeffs))
        y_double_prime = sum(coeff * basis_function_derivative(k, x_i, 2) for k, coeff in enumerate(coeffs))
        
        # Подставляем в уравнение
        equations[i] = x_i**2 * y_double_prime + x_i * y_prime - 16 * y - f(x_i)
    
    # Учитываем правое граничное условие y'(b) - y(b) = bc_b
    y_at_b = sum(coeff * basis_function(k, b) for k, coeff in enumerate(coeffs))
    y_prime_at_b = sum(coeff * basis_function_derivative(k, b, 1) for k, coeff in enumerate(coeffs))
    equations[n] = y_prime_at_b - y_at_b - bc_b
    
    return equations

# Решение системы методом коллокаций
def solve_collocation_method():
    # Начальное приближение для коэффициентов (n+1 коэффициентов для полинома степени n)
    initial_coeffs = np.zeros(n+1)
    # print(fsolve(collocation_system, initial_coeffs, full_output=True))
    # Решаем систему нелинейных уравнений
    coeffs = fsolve(collocation_system, initial_coeffs)
    print(coeffs)
    # if info['fvec'].max() > 1e-10:
    #     print("Предупреждение: возможно, решение не найдено с достаточной точностью")
    
    # Возвращаем функцию приближенного решения
    def approximate_solution(x):
        result = sum(coeff * basis_function(k, x) for k, coeff in enumerate(coeffs))
        return result
    
    # Возвращаем также производную приближенного решения
    def approximate_derivative(x):
        result = sum(coeff * basis_function_derivative(k, x, 1) for k, coeff in enumerate(coeffs))
        return result
    
    return approximate_solution, approximate_derivative, coeffs

# Проверка граничных условий
def check_boundary_conditions(approx_solution, approx_derivative):
    y_at_a = approx_solution(a)
    y_prime_minus_y_at_b = approx_derivative(b) - approx_solution(b)
    
    print(f"Граничное условие в x = {a}: y({a}) = {y_at_a:.6f} (должно быть {y_a})")
    print(f"Граничное условие в x = {b}: y'({b}) - y({b}) = {y_prime_minus_y_at_b:.6f} (должно быть {bc_b})")

# Вычисление погрешности
def calculate_error(approx_solution, x_points):
    y_numerical = np.array([approx_solution(x) for x in x_points])
    y_exact = exact_solution(x_points)
    
    absolute_error = np.abs(y_numerical - y_exact)
    relative_error = absolute_error / (np.abs(y_exact) + 1e-10) * 100  # в процентах
    
    max_abs_error = np.max(absolute_error)
    max_rel_error = np.max(relative_error)
    std = (np.sum((y_numerical - y_exact)**2) / 7) ** 0.5
    
    print("\nПогрешности:")
    print(f"Максимальная абсолютная погрешность: {max_abs_error:.6f}")
    print(f"Максимальная относительная погрешность: {max_rel_error:.6f}%")
    print(f"Среднее отклонение: {std:.6f}")
    
    return absolute_error, relative_error, y_numerical

# Построение графика
def plot_results(x_points, y_numerical, coeffs):
    # Генерируем больше точек для гладкого графика
    x_plot = np.linspace(a, b, 100)
    
    # Точное решение
    y_exact = exact_solution(x_plot)
    
    # Приближенное решение для точек графика
    y_approx = np.array([sum(coeff * basis_function(k, x) for k, coeff in enumerate(coeffs)) for x in x_plot])
    
    # Значения в узлах коллокации для отображения на графике
    y_exact_collocation = exact_solution(x_points)
    
    
    # График решений
    # plt.scatter(x_plot, y_exact, 'b-', label='Точное решение')
    plt.plot(x_plot, y_approx, 'r-', color='yellow', label='Приближенное решение (метод коллокаций)')
    # plt.scatter(x_points, y_numerical, color='red', marker='o', label='Узлы коллокации')
    plt.scatter(x_plot[::2], y_exact[::2], color='black', marker='*', label='Точное решение')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Сравнение приближенного и точного решений')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("./labs2/num_acc.jpg")
    
    # Вывод полиномиальной аппроксимации
    print("\nПолиномиальная аппроксимация:")
    poly = Polynomial(coeffs)
    print(f"P(x) = {poly}")

# Основная функция
def main():
    print("Решение краевой задачи методом коллокаций")
    print(f"Уравнение: x^2 y'' + xy' - 16y = -16 ln(x)")
    print(f"Граничные условия: y({a}) = {y_a}, y'({b}) - y({b}) = {bc_b}")
    print(f"Точное решение: y = x^4 + ln(x)")
    print(f"Число узлов коллокации: {n}")
    
    # Получаем приближенное решение методом коллокаций
    approx_solution, approx_derivative, coeffs = solve_collocation_method()
    
    # Генерируем точки для отображения результатов
    x_points = np.linspace(a, b, 20)  # больше точек для лучшей визуализации
    
    # Вычисляем значения приближенного решения в этих точках
    y_numerical = np.array([approx_solution(x) for x in x_points])
    
    # Выводим результаты
    print("\nЗначения приближенного и точного решений в некоторых точках:")
    for i, x_i in enumerate(x_points):
        print(f"x = {x_i:.4f}, y_approx = {approx_solution(x_i):.6f}, y_exact = {exact_solution(x_i):.6f}")
    
    # Проверяем граничные условия
    check_boundary_conditions(approx_solution, approx_derivative)
    
    # Вычисляем погрешность
    absolute_error, relative_error, y_collocation = calculate_error(approx_solution, collocation_points)
    
    # Строим графики
    plot_results(collocation_points, y_collocation, coeffs)

if __name__ == "__main__":
    main() 