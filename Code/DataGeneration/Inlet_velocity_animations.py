import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation

from bloodflow_functions import vel_csv

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r')

U_0 = [[] for i in range(5)]
U0_vp5 = [[] for i in range(11)]
U0_vp6 = [[] for i in range(11)]

for vp in range(7):
    U0, U0_vp5, U0_vp6, t = vel_csv(vp, U_0, U0_vp5, U0_vp6)

trange = np.linspace(0.01, 1, 100)
writergif = animation.PillowWriter(fps=20)

def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel('velocity [m/s]')
    ax.set_xlabel('time [s]')
    return ln,

def update(frame):
    xdata = trange[:frame]
    ydata = y[:frame]
    ln.set_data(xdata, ydata)
    return ln,

for i in range(1):
    y = np.asarray(U0_vp5[9])
    y = y[1:]
    ani = animation.FuncAnimation(fig, update, frames=len(trange), interval = 1,
                    init_func=init, blit=False)

    fig.suptitle(f'Velocity pattern 5.{10+i}')
    ani.save(f'ani_vp5.{10+i}.gif', writer=writergif)
    plt.show()