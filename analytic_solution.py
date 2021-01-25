import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange

u0 = 5
u = np.empty([101, 101])

x_list = np.linspace(0, 1, 101)
t_list = np.linspace(0, 1, 101)
dt = t_list[-1]/(len(t_list)-1)

@jit
def eval_analytic_1d(u):
    for n in prange(0, 101):
        for i in prange(0, 101):
            u[n,i] = u0 * (1-x_list[i]) + (2*u0/np.pi) * np.sin(np.pi*x_list[i]) * np.exp(-np.pi**2 * t_list[n])
    return u

eval_analytic_1d(u)

# 2D plots
# Note: time must satisfy condition: t >= 1/pi^2 for reasons left as an exercise
# for the reader
fig = plt.figure(figsize=(8,6))
num_of_slice = 10
plt.axhline(y=5)
for i in range(10, len(u), len(u)//num_of_slice ):
    plt.plot(x_list, u[i], label=f't={i*dt :.1g}')

plt.title('1d heat diffusion analytic')
plt.xlabel('Space domain')
plt.ylabel('Temperature')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# 3D plots
'''
x_list, t_list = np.meshgrid(x_list,t_list)
print(f'size of t_list: {t_list.shape}, size of x_list: {x_list.shape}, size of u: {u.shape}')

fig = plt.figure(figsize=(10,5))
ax = plt.axes(projection='3d')

ax.plot_wireframe(x_list, t_list, u, rcount=t_list.shape[1], ccount=1)
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('T')
ax.set_title('3D contour')
plt.show()
'''

# 3D contour
'''
plt.contourf(x_list, t_list, u)
plt.show()
'''