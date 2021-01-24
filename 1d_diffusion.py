import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from mpl_toolkits.mplot3d import Axes3D
# Forward explicit, 1d heat diffusion, diffusivity = 1
# Boundary condition: f(x=0, t) = 5, f(x=L) = 0
# Initial condition: f(x, t=0) = 0

# Set the dimensions
# L -> length of the space domain
# I -> number of space subdivisions
# dx -> space step
# N -> number of time subdivisions
# expiration_time -> end time
# dt -> time step

L = 1.0
I = 100 # for i = 0, 1, ..., I -> 101 domain points
dx = L / I 
expiration_time = 0.5
dt = 0.5 * dx**2 # For stability reasons
N = int(expiration_time / dt) # for n = 0, 1, ..., N -> N+1 time points
print(f'domain step: {dx :.3f}; time step: {dt :.5f}; number of domain divisions: {I}; number of time subdivisions: {N}')

# Set the Boundary conditions and the initiail value function
# Initialize solution: the grid of u(n, i)
u = np.empty([N+1, I+1])

# Initial condition everywhere inside the grid
u_initial = 0

# Boundary conditions
u_left = 5.0
u_right = 0.0

# Set the initial condition
u.fill(u_initial)

# Set the boundary conditions
u[:, 0] = u_left
u[:, -1] = u_right

@jit 
def eval_1d(u):
    for n in prange(0, N):
        for i in prange(1, I):
            u[n+1,i] = u[n,i] + (dt/dx**2) * (u[n,i+1] - 2*u[n,i] + u[n,i-1])
    return u

eval_1d(u)
# 2D plots
'''
x_list = np.linspace(0, 1, I+1)
t_list = np.linspace(0, 1, N+1)
plt.plot(x_list, u[0], color='red')
plt.plot(x_list, u[1000], color='green')
plt.plot(x_list, u[2500], color='green')
plt.plot(x_list, u[5000], color='green')
plt.plot(x_list, u[-1], color='blue')
plt.title('1d heat diffusion')
plt.xlabel('X range')
plt.ylabel('Temperature')
plt.show()
'''
# 3D plots

t_list = np.linspace(0, expiration_time, u.shape[0])
x_list = np.linspace(0, L, u.shape[1])
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

# 3D contour
'''
plt.contourf(x_list, t_list, u)
plt.show()
'''