import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd 


# Начальные параметры
x_start = 0
x_end = 1
t_start = 0
t_end = 0.5


h = 0.1
Nx = int((x_end - x_start) / h) + 1
x = np.linspace(x_start, x_end, Nx)

# Выбираем шаг по времени согласно условию устойчивости Куранта-Фридрихса-Леви
# Для явной схемы должно выполняться условие tau <= h

tau = 0.05  # Выбираем tau = h/2 для устойчивости
# Количество точек по t
Nt = int((t_end - t_start) / tau) + 1
# Точки по t
t = np.linspace(t_start, t_end, Nt)

# Инициализация сетки решения
u = np.zeros((Nx, Nt))

# Начальное условие по x при t = 0: u(x, 0) = 0.5*x*(x+1)
for i in range(Nx):
    u[i, 0] = 0.5 * x[i] * (x[i] + 1)

# Краевые условия по t
for j in range(Nt):
    u[0, j] = 2 * t[j]**2  # u(0, t) = 2t^2
    u[Nx-1, j] = 1  # u(1, t) = 1

# Вычисление значений для первого временного слоя (j = 1) с использованием начального условия по скорости
# Используем аппроксимацию: u(x, tau) ≈ u(x, 0) + tau * u_t'(x, 0) + (tau^2/2) * u_tt''(x, 0)
# Где u_tt''(x, 0) можно найти из уравнения: u_tt''(x, 0) = u_xx''(x, 0)

for i in range(1, Nx-1):
    u_t_initial = x[i] * np.cos(x[i])
    
    u_xx_initial = (u[i+1, 0] - 2*u[i, 0] + u[i-1, 0]) / (h**2)
    
    u[i, 1] = u[i, 0] + tau * u_t_initial + (tau**2 / 2) * u_xx_initial

# Основной цикл решения методом сеток (явная схема)
for j in range(1, Nt-1):
    for i in range(1, Nx-1):
        u[i, j+1] = 2*u[i, j] - u[i, j-1] + gamma**2 * (u[i+1, j] - 2*u[i, j] + u[i-1, j])


print("Решение уравнения колебания струны методом сеток")
print("Параметры: h =", h, "tau =", tau, "gamma =", gamma)


print("\nНачальная функция u(x, 0) = 0.5*x*(x+1):")
for i in range(Nx):
    print(f"x = {x[i]:.1f}, u(x, 0) = {u[i, 0]:.4f}")


print("\nРешение u(x, t):")

data = {}
for j in range(0, Nt, 2):  
    data[f"t = {t[j]:.2f}"] = u[:, j] 


# df = pd.DataFrame(data, index=[f"{xi:.1f}" for xi in x])
# df.index.name = "x"

# print(df.to_latex())

print("    x  |", end="")
for j in range(0, Nt, 2):  # Выводим не все моменты времени для компактности
    print(f"  t = {t[j]:.2f}  |", end="")
print()
print("------|" + "------------|" * (len(range(0, Nt, 2))))

for i in range(Nx):
    print(f" {x[i]:.1f}  |", end="")
    for j in range(0, Nt, 2):  # Выводим не все моменты времени для компактности
        print(f"  {u[i, j]:.4f}   |", end="")
    print()

    
# Создание графика 3D поверхности
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


X, T = np.meshgrid(x, t)
Z = u.T
surf = ax.plot_surface(X, T, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)


fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Подписи осей
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('Решение уравнения колебания струны')

# plt.show()


