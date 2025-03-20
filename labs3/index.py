import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import fsolve
import sympy as sp
import time
from functools import lru_cache

# Измерение общего времени выполнения
start_time = time.time()

# Параметры задачи
l = 1.2    # длина балки, м
q0 = 1200  # интенсивность нагрузки
EI = 2e4   # жесткость балки

# Символьная переменная для аналитических выражений
x = sp.symbols('x')

# Определение нагрузки
q_symbolic = q0 * (sp.sin((sp.pi * x) / l + (x / l) ** 2))

# Преобразование символьного выражения в числовую функцию
q_numpy = sp.lambdify(x, q_symbolic, 'numpy')

print("Инициализация параметров: {:.2f} сек".format(time.time() - start_time))
time_checkpoint = time.time()

# =========== ЧИСЛЕННЫЙ МЕТОД ДЛЯ ТОЧНОГО РЕШЕНИЯ ===========

# Метод конечных разностей
def solve_beam_finite_difference(n_points=201):
    """
    Решение дифференциального уравнения изгиба балки методом конечных разностей
    """
    # Дискретизация балки
    x_values = np.linspace(0, l, n_points)
    dx = l / (n_points - 1)
    
    # Матрица для системы конечных разностей (левая часть уравнения EI*w'''' = q)
    A = np.zeros((n_points, n_points))
    
    # Вектор правой части (нагрузка)
    b = np.zeros(n_points)
    
    # Заполнение матрицы для внутренних точек (аппроксимация 4-й производной)
    for i in range(2, n_points-2):
        A[i, i-2:i+3] = np.array([1, -4, 6, -4, 1]) / (dx**4)
        # Правая часть: q(x)/EI
        if x_values[i] <= l/2:  # Нагрузка только на первой половине
            b[i] = q_numpy(x_values[i]) / EI
    
    # Граничные условия
    # При x=0: w=0, w'=0 (жесткая заделка)
    A[0, :] = 0
    A[0, 0] = 1  # w(0) = 0
    b[0] = 0
    
    A[1, :] = 0
    A[1, 0:3] = np.array([-3, 4, -1]) / (2*dx)  # Аппроксимация w'(0) = 0
    b[1] = 0
    
    # При x=l: w=0, w'=0 (свободное опирание)
    A[-1, :] = 0
    A[-1, -1] = 1  # w(l) = 0
    b[-1] = 0
    
    A[-2, :] = 0
    A[-2, -3:] = np.array([1, -4, 3]) / (2*dx)  # Аппроксимация w'(l) = 0
    b[-2] = 0
    
    # Решение системы
    try:
        w = np.linalg.solve(A, b)
        
        # Вычисление момента (пропорционален второй производной)
        M = np.zeros(n_points)
        for i in range(1, n_points-1):
            M[i] = EI * (w[i-1] - 2*w[i] + w[i+1]) / (dx**2)
        
        # Вычисление перерезывающей силы (пропорциональна третьей производной)
        Q = np.zeros(n_points)
        for i in range(1, n_points-2):
            Q[i] = -EI * (w[i-2] - 2*w[i-1] + 2*w[i+1] - w[i+2]) / (2*dx**3)
            
        return x_values, w, M, Q
    except np.linalg.LinAlgError:
        print("Ошибка решения системы уравнений. Возможно, матрица вырождена.")
        return None, None, None, None

# Решение методом конечных разностей
x_fd, w_fd, M_fd, Q_fd = solve_beam_finite_difference(201)

print("Решение методом конечных разностей: {:.2f} сек".format(time.time() - time_checkpoint))
time_checkpoint = time.time()

# =========== ОПТИМИЗИРОВАННЫЙ МЕТОД РИТЦА ===========

# Базисные функции для метода Ритца
@lru_cache(maxsize=None)
def phi_1(x_val):
    """Первая базисная функция: sin²(πx/l)"""
    return np.sin(np.pi * x_val / l)**2

@lru_cache(maxsize=None)
def phi_2(x_val):
    """Вторая базисная функция: x²(1-x/l)²"""
    return x_val**2 * (1 - x_val/l)**2

@lru_cache(maxsize=None)
def d2phi_1(x_val):
    """Вторая производная первой базисной функции"""
    return -((np.pi/l)**2) * np.sin(np.pi * x_val / l) * (2 * np.cos(np.pi * x_val / l))

@lru_cache(maxsize=None)
def d2phi_2(x_val):
    """Вторая производная второй базисной функции"""
    return 2 * (1 - x_val/l) * (1 - 3*x_val/l) + 2*x_val/l * (2 - 3*x_val/l)

# Численное интегрирование для метода Ритца
def compute_ritz_coefficients():
    # Потенциальная энергия деформации: U = (EI/2) ∫(w''(x))² dx
    def U_integrand(x_val, a1, a2):
        d2w = a1 * d2phi_1(x_val) + a2 * d2phi_2(x_val)
        return EI/2 * d2w**2
    
    # Работа внешних сил: V = ∫q(x)·w(x)dx
    def V_integrand(x_val, a1, a2):
        if x_val <= l/2:  # Нагрузка только на первой половине
            w_val = a1 * phi_1(x_val) + a2 * phi_2(x_val)
            return q_numpy(x_val) * w_val
        return 0
    
    # Матрица и вектор для системы a·K = F
    K = np.zeros((2, 2))
    F = np.zeros(2)
    
    # Вычисление компонентов матрицы жесткости K
    for i in range(2):
        for j in range(2):
            if i == 0 and j == 0:
                # ∫(phi_1''(x))²dx
                def integrand(x_val):
                    return EI/2 * d2phi_1(x_val)**2
            elif i == 0 and j == 1:
                # ∫phi_1''(x)·phi_2''(x)dx
                def integrand(x_val):
                    return EI * d2phi_1(x_val) * d2phi_2(x_val)
            elif i == 1 and j == 0:
                # ∫phi_2''(x)·phi_1''(x)dx
                def integrand(x_val):
                    return EI * d2phi_2(x_val) * d2phi_1(x_val)
            else:  # i == 1 and j == 1
                # ∫(phi_2''(x))²dx
                def integrand(x_val):
                    return EI/2 * d2phi_2(x_val)**2
            
            K[i, j] = integrate.quad(integrand, 0, l)[0]
    
    # Вычисление вектора нагрузки F
    for i in range(2):
        if i == 0:
            # ∫q(x)·phi_1(x)dx
            def integrand(x_val):
                if x_val <= l/2:
                    return q_numpy(x_val) * phi_1(x_val)
                return 0
        else:  # i == 1
            # ∫q(x)·phi_2(x)dx
            def integrand(x_val):
                if x_val <= l/2:
                    return q_numpy(x_val) * phi_2(x_val)
                return 0
        
        F[i] = integrate.quad(integrand, 0, l/2)[0]
    
    # Решение системы K·a = F
    a = np.linalg.solve(K, F)
    
    return a[0], a[1]

# Вычисление коэффициентов Ритца
a1_numeric, a2_numeric = compute_ritz_coefficients()

print("Метод Ритца (численное интегрирование): {:.2f} сек".format(time.time() - time_checkpoint))
time_checkpoint = time.time()

print(f"Коэффициенты Ритца: a1 = {a1_numeric:.6f}, a2 = {a2_numeric:.6f}")

# Функции для вычисления приближенного решения методом Ритца
def w_ritz(x_val):
    """Функция прогиба (метод Ритца)"""
    return a1_numeric * phi_1(x_val) + a2_numeric * phi_2(x_val)

def M_ritz(x_val):
    """Изгибающий момент (метод Ритца)"""
    return EI * (a1_numeric * d2phi_1(x_val) + a2_numeric * d2phi_2(x_val))

def Q_ritz(x_val):
    """Перерезывающая сила (метод Ритца)"""
    # Численное дифференцирование для третьей производной
    dx = 1e-6
    return (M_ritz(x_val + dx) - M_ritz(x_val)) / dx

# =========== ВЫЧИСЛЕНИЕ РЕЗУЛЬТАТОВ И ВИЗУАЛИЗАЦИЯ ===========

# Массив точек для построения графиков
x_values = np.linspace(0, l, 100)

# Вычисление прогиба, момента и перерезывающей силы методом Ритца
w_ritz_values = np.array([w_ritz(x_val) for x_val in x_values])
M_ritz_values = np.array([M_ritz(x_val) for x_val in x_values])
Q_ritz_values = np.array([Q_ritz(x_val) for x_val in x_values])

# Функция для вычисления среднеквадратического отклонения
def relative_error(approximate, exact):
    """Вычисляет относительное среднеквадратическое отклонение"""
    valid_indices = ~np.isnan(exact) & ~np.isnan(approximate)
    if not np.any(valid_indices):
        return float('nan')
    
    exact_valid = exact[valid_indices]
    approx_valid = approximate[valid_indices]
    
    if np.all(np.abs(exact_valid) < 1e-10):
        return 0.0  # Избегаем деления на ноль
        
    summa1 = np.sum((approx_valid - exact_valid)**2)
    summa2 = np.sum(exact_valid**2)
    return np.sqrt(summa1 / summa2) * 100  # В процентах

# Интерполяция результатов метода конечных разностей на равномерную сетку
from scipy.interpolate import interp1d

if w_fd is not None:
    w_fd_interp = interp1d(x_fd, w_fd, kind='cubic', bounds_error=False, fill_value=np.nan)
    M_fd_interp = interp1d(x_fd, M_fd, kind='cubic', bounds_error=False, fill_value=np.nan)
    Q_fd_interp = interp1d(x_fd, Q_fd, kind='cubic', bounds_error=False, fill_value=np.nan)
    
    w_fd_values = w_fd_interp(x_values)
    M_fd_values = M_fd_interp(x_values)
    Q_fd_values = Q_fd_interp(x_values)
    
    # Вычисляем среднеквадратическое отклонение
    w_error = relative_error(w_ritz_values, w_fd_values)
    M_error = relative_error(M_ritz_values, M_fd_values)
    Q_error = relative_error(Q_ritz_values, Q_fd_values)
    
    print(f"Среднеквадратическое отклонение для прогиба W(x): {w_error:.2f}%")
    print(f"Среднеквадратическое отклонение для момента M(x): {M_error:.2f}%")
    print(f"Среднеквадратическое отклонение для перерезывающей силы Q(x): {Q_error:.2f}%")
else:
    print("Невозможно вычислить ошибки, так как точное решение не было получено.")

print("Вычисление результатов и ошибок: {:.2f} сек".format(time.time() - time_checkpoint))
time_checkpoint = time.time()

# Построение графиков
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# График прогиба
axs[0, 0].plot(x_values, w_ritz_values, 'r--', linewidth=2, label='Метод Ритца')
if w_fd is not None:
    axs[0, 0].plot(x_values, w_fd_values, 'b-', linewidth=1.5, label='Метод конечных разностей')
axs[0, 0].set_xlabel('x, м')
axs[0, 0].set_ylabel('w(x), м')
axs[0, 0].set_title('Функция прогиба w(x)')
axs[0, 0].legend()
axs[0, 0].grid(True)

# График момента
axs[0, 1].plot(x_values, M_ritz_values, 'r--', linewidth=2, label='Метод Ритца')
if M_fd is not None:
    axs[0, 1].plot(x_values, M_fd_values, 'b-', linewidth=1.5, label='Метод конечных разностей')
axs[0, 1].set_xlabel('x, м')
axs[0, 1].set_ylabel('M(x), Н·м')
axs[0, 1].set_title('Изгибающий момент M(x)')
axs[0, 1].legend()
axs[0, 1].grid(True)

# График перерезывающей силы
axs[1, 0].plot(x_values, Q_ritz_values, 'r--', linewidth=2, label='Метод Ритца')
if Q_fd is not None:
    axs[1, 0].plot(x_values, Q_fd_values, 'b-', linewidth=1.5, label='Метод конечных разностей')
axs[1, 0].set_xlabel('x, м')
axs[1, 0].set_ylabel('Q(x), Н')
axs[1, 0].set_title('Перерезывающая сила Q(x)')
axs[1, 0].legend()
axs[1, 0].grid(True)

# График нагрузки
q_values = np.zeros_like(x_values)
for i, x_val in enumerate(x_values):
    if x_val <= l/2:
        q_values[i] = q_numpy(x_val)

axs[1, 1].plot(x_values, q_values, 'g-', linewidth=2)
axs[1, 1].set_xlabel('x, м')
axs[1, 1].set_ylabel('q(x), Н/м')
axs[1, 1].set_title('Распределенная нагрузка q(x)')
axs[1, 1].grid(True)

plt.savefig("balk.png")
plt.tight_layout()

print("Построение графиков: {:.2f} сек".format(time.time() - time_checkpoint))
print("Общее время выполнения: {:.2f} сек".format(time.time() - start_time))

plt.show()