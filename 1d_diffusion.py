import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
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
print(f'domain step: {dx :.3f}; time step: {dt :.5f}; number of time subdivisions: {N}')

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
    for n in prange(0, N+1):
        for i in prange(1, I+1):
            u[n+1,i] = u[n,i] + (dt/dx**2) * (u[n,i+1] - 2*u[n,i] + u[n,i-1])
    return u

eval_1d(u)
print(u[10000])
x_list = np.linspace(0, 1, I+1)
plt.plot(x_list, u[0], color='red')
plt.plot(x_list, u[1000], color='green')
plt.plot(x_list, u[2500], color='green')
plt.plot(x_list, u[5000], color='green')
plt.plot(x_list, u[-1], color='blue')
plt.title('1d heat diffusion')
plt.xlabel('X range')
plt.ylabel('Temperature')
plt.show()
