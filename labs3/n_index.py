import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


l = 1.2    # длина балки, м
q0 = 1200  # интенсивность нагрузки
EI = 2e4   # жесткость балки

# Символьная переменная для аналитических выражений
x = sp.symbols('x')
a1, a2 = sp.symbols('a1 a2')

# Определение нагрузки
q = q0 * (sp.sin((sp.pi * x) / l + (x / l) ** 2))


# Функции прогиба и их производные для двух участков балки
def w1(x, c1, c2, c3, c4):
    """Функция прогиба для первой половины балки (0 ≤ x ≤ l/2)"""
    return 3 * q0 / EI * (-4/np.pi**5 * np.sin(np.pi*x) +
                          1/np.pi**4 * x * np.cos(np.pi*x) +
                          c1/6 * x**3 + c2/2 * x**2 + c3*x + c4)

def w2(x, d1, d2, d3, d4):
    """Функция прогиба для второй половины балки (l/2 ≤ x ≤ l)"""
    return (1 / EI) * (x**3 / 6 * d1 + x**2 / 2 * d2 + x * d3 + d4)

def w1_prime(x, c1, c2, c3):
    """Первая производная прогиба для первой половины"""
    return (3 * q0 / EI) * (-3/np.pi**4 * np.cos(np.pi*x) -
                            1/np.pi**3 * x * np.sin(np.pi*x) +
                            c1/2 * x**2 + c2 * x + c3)

def w2_prime(x, d1, d2, d3):
    """Первая производная прогиба для второй половины"""
    return (1 / EI) * (x**2 / 2 * d1 + x * d2 + d3)

def w1_double_prime(x, c1, c2):
    """Вторая производная прогиба для первой половины"""
    return (3 * q0 / EI) * (2/np.pi**3 * np.sin(np.pi*x) -
                            1/np.pi**2 * x * np.cos(np.pi*x) + c1 * x + c2)

def w2_double_prime(x, d1, d2):
    """Вторая производная прогиба для второй половины"""
    return (1 / EI) * (x * d1 + d2)

def w1_triple_prime(x, c1):
    """Третья производная прогиба для первой половины"""
    return (3 * q0 / EI) * (1/np.pi**2 * np.cos(np.pi*x) +
                            1/np.pi * x * np.sin(np.pi*x) + c1)

def w2_triple_prime(d1):
    """Третья производная прогиба для второй половины"""
    return (1 / EI) * d1



# Система уравнений для нахождения констант
def equations(vars):
    c1, c2, c3, c4, d1, d2, d3, d4 = vars
    eqs = [
        # Граничные условия
        w1(0, c1, c2, c3, c4),  # w1(0) = 0
        w1_prime(0, c1, c2, c3),  # w1'(0) = 0
        w2(l, d1, d2, d3, d4),  # w2(l) = 0
        w2_prime(l, d1, d2, d3),  # w2'(l) = 0
        # Условия совместимости в точке l/2
        w1(l/2, c1, c2, c3, c4) - w2(l/2, d1, d2, d3, d4),
        w1_prime(l/2, c1, c2, c3) - w2_prime(l/2, d1, d2, d3),
        w1_double_prime(l/2, c1, c2) - w2_double_prime(l/2, d1, d2),
        w1_triple_prime(l/2, c1) - w2_triple_prime(d1),
    ]
    return eqs


initial_guess = np.zeros(8)

# Решение системы уравнений для коэффициентов c1, c2, c3, c4, d1, d2, d3, d4
solution = fsolve(equations, initial_guess)
c1, c2, c3, c4, d1, d2, d3, d4 = solution

# Вывод значений констант
print("Константы интегрирования: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(c1, c2, c3, c4, d1, d2, d3, d4))

# Базисные функции
phi_1 = sp.sin(sp.pi * x / l)**2
phi_2 = x**2 * (1 - x / l)**2

# Функция прогиба
w_ritz = a1 * phi_1 + a2 * phi_2

# Вторая производная
d2w_ritz = sp.diff(w_ritz, x, 2)
# Потенциальная энергия
v = EI / 2 * sp.integrate(d2w_ritz**2, (x, 0, l))
u = sp.integrate(q * w_ritz, (x, 0, l/2))

eq1 = sp.diff(v - u, a1)
eq2 = sp.diff(v - u, a2)

# Решение системы уравнений для коэффициентов a1, a2
solution = sp.solve([eq1, eq2], (a1, a2))

# Подстановка найденных коэффициентов в функцию прогиба
w_ritz_solution = w_ritz.subs({a1: solution[a1], a2: solution[a2]})
w_ritz_solution_rounded = sp.N(w_ritz_solution, 7)

print("Функция прогиба (метод Ритца):")
sp.pprint(w_ritz_solution_rounded)


x_values_exact = np.linspace(0, l, 100)

w_values_exact = np.where(x_values_exact <= l/2,
                          w1(x_values_exact, c1, c2, c3, c4),
                          w2(x_values_exact, d1, d2, d3, d4))

M_values_exact = np.where(x_values_exact <= l/2,
                          EI * w1_double_prime(x_values_exact, c1, c2),
                          EI * w2_double_prime(x_values_exact, d1, d2))

Q_values_exact = np.where(x_values_exact <= l/2,
                          EI * w1_triple_prime(x_values_exact, c1),
                          EI * w2_triple_prime(d1))

# Момент и перерезывающая сила для метода Ритца
M_ritz = EI * sp.diff(w_ritz_solution, x, 2)
Q_ritz = EI * sp.diff(w_ritz_solution, x, 3)

# Преобразуем символьные функции в числовые для построения графиков
w_ritz_func = sp.lambdify(x, w_ritz_solution, 'numpy')
M_ritz_func = sp.lambdify(x, M_ritz, 'numpy')
Q_ritz_func = sp.lambdify(x, Q_ritz, 'numpy')


# Вычисление максимальной абсолютной погрешности
def cko(ritz_func, values_exact):
    summa1 = np.sum((ritz_func - values_exact)**2)
    summa2 = np.sum(values_exact**2)
    return np.sqrt(summa1) / np.sqrt(summa2)


# Вычисляем среднеквадратическое отклонение
print("Среднеквадратическое отклонение для прогиба W(x): {}".format(cko(w_ritz_func(x_values_exact), w_values_exact) * 100))
print("Среднеквадратическое отклонение для момента M(x): {}".format(cko(M_ritz_func(x_values_exact), M_values_exact) * 100))
print("Среднеквадратическое отклонение для перерезывающей силы Q(x): {}".format(cko(Q_ritz_func(x_values_exact), Q_values_exact) * 100))

# Построение графиков
plt.figure(figsize=(15, 5))
# График прогиба
plt.subplot(2, 2, 1)
plt.plot(x_values_exact, w_ritz_func(x_values_exact),
         label='Приближённое решение')
plt.plot(x_values_exact, w_values_exact, label='Точное решение')
plt.xlabel('x, м')
plt.ylabel('w(x), м')
plt.legend()
plt.grid()
plt.title('Функция прогиба w(x)')

# График момента
plt.subplot(2, 2, 2)
plt.plot(x_values_exact, M_ritz_func(x_values_exact),
         label='Приближённое решение')
plt.plot(x_values_exact, M_values_exact, label='Точное решение')
plt.xlabel('x, м')
plt.ylabel('M(x), кг*м')
plt.legend()
plt.grid()
plt.title('Момент M(x)')

# График перерезывающей силы
plt.subplot(2, 2, 3)
plt.plot(x_values_exact, Q_ritz_func(x_values_exact),
         label='Приближённое решение')
plt.plot(x_values_exact, Q_values_exact, label='Точное решение')
plt.xlabel('x, м')
plt.ylabel('Q(x), кг')
plt.legend()
plt.grid()
plt.title('Перерезывающая сила Q(x)')

plt.tight_layout()
plt.show()