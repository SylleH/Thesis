"""
Author: Sylle Hoogeveen
Functions to create velocity inlet and pressure outlet functions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from matplotlib import animation



delta_T = 0.01
t = np.linspace(0,1,int(1/delta_T)+1) #one cycle

def test_func(x, a, b, c, d, e,f,g ,h):
    return a*np.sin(b*x) + c*np.sin(d*x) + e*np.sin(f*x) + g*np.sin(h*x)

def v_in(t):
    """
    function to create time dependent velocity inlet boundary condition
    velocity pulse from t=0 to t=0.5, constant from t=0.5 to t=1
    :param t: time vector
    :return v_list: velocity list containing corresponding velocity values
    """
    y = [0.08,0.21,0.62,0.53,0.37,0.22,0.02,0.05,0.07,0.08, 0.07, 0.08,  0.07, 0.08, 0.08]
    x = np.linspace(0,0.75,15)

    v_coef = np.polyfit(x,y,5)
    p6 = np.poly1d(v_coef)
    xp6 = np.linspace(0,0.8,500)

    vx_list = []
    vy_list = []
    vz_list = []

    v_list = [] #for plotting
    for count, value in enumerate(t):
        if value<0.8:
            v_list.append(p6(value))
            vy_list.append(-p6(value))
        else:
            v_list.append(p6(0.8))
            vy_list.append(p6(0.8))
        vx_list.append(0)
        vz_list.append(0)

    print(v_list)
    t_list = t.tolist()
    v_dict = {'time': t_list, 'velocity_x': vx_list, 'velocity_y': vy_list, 'velocity_z': vz_list}

    plt.plot(x,y, '.',xp6,p6(xp6), '-')
    plt.show()
    return v_list, v_dict

def v_sine(U_max, U_min, tb, td,U_two=0, tp=0):
    t = np.arange(0,1.01,0.01)

    U_0 = np.zeros_like(t)
    Udif1 = np.zeros_like(t)
    Udif2 = np.zeros_like(t)
    Udif3 = np.zeros_like(t)
    Udif4 = np.zeros_like(t)
    Udif5 = np.zeros_like(t)
    # Udif6 = np.zeros_like(t)
    # Udif7 = np.zeros_like(t)
    # Udif8 = np.zeros_like(t)
    # Udif9 = np.zeros_like(t)
    # Udif10 = np.zeros_like(t)
    for count, value in enumerate(t):
        if (value<=tb):
            U_0[count] = 0.08 +U_max*(1-np.power(value-(tb/2),2)/((tb/2)*(tb/2)))
        if td:
            if ((value>tb) and (value<=(tb+td))):
                U_0[count] = 0.08 +U_min*(-1+np.power(value-(tb+td/2),2)/((td/2)*(td/2)))
        if tp:
            if ((value > (tb+td)) and (value <= (tb+td+tp))):
                U_0[count] = 0.08 + U_two * (1 - np.power(value - (tb+td+tp/2), 2) / ((tp / 2) * (tp / 2)))
        if (value>(tb+td+tp)):
            U_0[count] = 0.08
    for i in range(96):
        Udif1[i] = U_0[i+1] - U_0[i]
        Udif2[i] = U_0[i+2] - U_0[i]
        Udif3[i] = U_0[i+3] - U_0[i]
        Udif4[i] = U_0[i+4] - U_0[i]
        Udif5[i] = U_0[i+5] - U_0[i]
        # Udif6[i] = U_0[i+6] - U_0[i]
        # Udif7[i] = U_0[i+7] - U_0[i]
        # Udif8[i] = U_0[i+8] - U_0[i]
        # Udif9[i] = U_0[i+9] - U_0[i]
        # Udif10[i] = U_0[i+10] - U_0[i]
    return U_0, Udif1, Udif2, Udif3, Udif4, Udif5 #, Udif6, Udif7, Udif8, Udif9, Udif10

def vel_csv(vp, U_0, U0_vp5, U0_vp6):
    print(vp)
    t = np.arange(0,101,1)

    if vp == 0:
        Umax = 0.4
        Umin = 0.02
        tb = 0.35
        td = 0.25
    if vp == 1:
        Umax = 0.4
        Umin = 0
        tb = 0.1
        td = 0.1
    if vp == 2:
        Umax = 0.4
        Umin = 0
        tb = 0.6
        td = 0.1
    if vp == 3:
        Umax = 0.4
        Umin = 0.08
        tb = 0.2
        td = 0.1
    if vp == 4:
        Umax = 0.3
        Umin = 0.04
        tb = 0.3
        td = 0.26
    if vp == 5:
        Umax = [0.201, 0.292, 0.277, 0.351, 0.321, 0.353, 0.379, 0.358, 0.211, 0.366, 0.204]
        Umin = [0.038, 0.078, 0.011, 0.038, 0.054, 0.001, 0.025, 0.071, 0.069, 0.044, 0.043]
        tb = [0.27, 0.26, 0.25, 0.21, 0.3, 0.4, 0.25, 0.53, 0.11, 0.12, 0.23]
        td = [0.4, 0.4, 0.05, 0.26, 0.02, 0.06, 0.03, 0.31, 0.21, 0.23, 0.27]
    if vp == 6:
        Umax = [0.3, 0.211, 0.343, 0.221, 0.328, 0.368, 0.255, 0.218, 0.394, 0.29, 0.216]
        Umin = [0.002, 0.002, 0.012, 0.028, 0.04, 0.005, 0.063, 0.048, 0.013, 0.047,0.006]
        Utwo = [0.154, 0.094, 0.125, 0.121, 0.078, 0.174, 0.117, 0.181, 0.087,0.121,0.12]
        tb = [0.28, 0.15, 0.15, 0.1, 0.2, 0.13, 0.27, 0.14, 0.13, 0.12, 0.23]
        td = [0.27, 0.21, 0.21, 0.24, 0.21, 0.09, 0.26, 0.23, 0.13, 0.24, 0.29]
        tp = [0.2, 0.09, 0.17, 0.15, 0.06, 0.1, 0.21, 0.02, 0.17, 0.17, 0.16]
    if vp in [0,1,2,3,4]:
        U0, Udif1, Udif2, Udif3, Udif4, Udif5 = v_sine(Umax, Umin, tb, td) #, Udif6, Udif7, Udif8, Udif9, Udif10
        U_dict = {'time': t, 'U_0': U0, 'Udif1': Udif1, 'Udif2': Udif2, 'Udif3': Udif3,
                  'Udif4': Udif4, 'Udif5': Udif5}#, 'Udif6': Udif6, 'Udif7': Udif7, 'Udif8': Udif8,
                  #'Udif9': Udif9, 'Udif10': Udif10}
        U_df = pd.DataFrame(U_dict)
        U_df = U_df.round(5)
        U_df.to_csv(f"inlet_Us_diff/inlet_U_vp{vp}.csv", index=False)
        U_0[vp] = (list(U0))
    elif vp == 5:
        for i in range(11):
            U0, Udif1, Udif2, Udif3, Udif4, Udif5= v_sine(Umax[i], Umin[i], tb[i], td[i]) #, Udif6, Udif7, Udif8, Udif9, Udif10
            U_dict = {'time':t, 'U_0': U0, 'Udif1': Udif1, 'Udif2': Udif2, 'Udif3':Udif3,
                      'Udif4': Udif4, 'Udif5': Udif5}#, 'Udif6': Udif6, 'Udif7': Udif7, 'Udif8':Udif8,
                      #'Udif9': Udif9, 'Udif10': Udif10}
            U_df = pd.DataFrame(U_dict)
            U_df = U_df.round(5)
            U_df.to_csv(f"inlet_Us_diff/inlet_U_vp{vp}_{i+1}.csv", index=False)
            U0_vp5[i] = list(U0)
    else:
        for i in range(11):
            U0, Udif1, Udif2, Udif3, Udif4, Udif5 = v_sine(Umax[i], Umin[i], tb[i], td[i], Utwo[i], tp[i]) #, Udif6, Udif7, Udif8, Udif9, Udif10
            U_dict = {'time':t, 'U_0': U0, 'Udif1': Udif1, 'Udif2': Udif2, 'Udif3':Udif3,
                      'Udif4': Udif4, 'Udif5': Udif5 }#, 'Udif6': Udif6, 'Udif7': Udif7, 'Udif8':Udif8,
                      #'Udif9': Udif9, 'Udif10': Udif10}
            U_df = pd.DataFrame(U_dict)
            U_df = U_df.round(5)
            U_df.to_csv(f"inlet_Us_diff/inlet_U_vp{vp}_{i+1}.csv", index=False)
            U0_vp6[i] = list(U0)
    return U_0, U0_vp5, U0_vp6, t

def real_beat(a0, a, b):
    #a0 = 11.693284502463376
    a = np.asarray(a)
    b = np.asarray(b)
    # a = np.asarray([1.420706949636449, -0.937457438404759, 0.281479818173732, -0.224724363786734, 0.080426469802665,
    #                 0.032077024077824, 0.039516941555861, 0.032666881040235, -0.019948718147876, 0.006998975442773,
    #                 -0.033021060067630, -0.015708267688123,-0.029038419813160, -0.003001255512608, -0.009549531539299,
    #                 0.007112349455861, 0.001970095816773, 0.015306208420903, 0.006772571935245, 0.009480436178357])
    #
    # b = np.asarray([-1.325494054863285, 0.192277311734674, 0.115316087615845, -0.067714675760648, 0.207297536049255,
    #                 -0.044080204999886, 0.050362628821152, -0.063456242820606, \
    #                 -0.002046987314705, -0.042350454615554, -0.013150127522194, -0.010408847105535, 0.011590255438424,
    #                 0.013281630639807, 0.014991955865968, 0.016514327477078, \
    #                 0.013717154383988, 0.012016806933609, -0.003415634499995, 0.003188511626163])

    t_min = 0
    t_max = t_min + 1.0
    dt = 0.01
    n = int(t_max / dt)
    r = 1  # radius in cm
    A = np.pi * r ** 2
    Q = 0.5 * a0
    trange = np.linspace(start=t_min, stop=t_max, num=n + 1)
    x = np.zeros_like(trange)
    for k in range(len(trange)):
        x[k] = np.pi * (2 * (trange[k] - t_min) / (t_max - t_min) - 1)

    for i in range(10):
        Q += a[i] * np.cos((i + 1) * x) + b[i] * np.sin((i + 1) * x)
    #Q -= 2
    #Q -= 0.026039341343493
    v = Q / A #* 25  # flow to velocity in cm/s (25 also tried)
    v = v / 100  # velocity in m/s
    plt.plot(trange, v, 'b')
    plt.rcParams['font.size'] = '18'
    plt.xlim([0, 1])
    plt.ylim([-0.01, 0.6])
    plt.xlabel('time [s]')
    plt.ylabel('velocity [m/s]')
    plt.title('Velocity pattern realistic heartbeat')
    plt.tight_layout()
    plt.savefig('vp_realbeat_aorta.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    Udif1 = np.zeros_like(trange)
    Udif2 = np.zeros_like(trange)
    Udif3 = np.zeros_like(trange)
    Udif4 = np.zeros_like(trange)
    Udif5 = np.zeros_like(trange)
    for i in range(96):
        Udif1[i] = v[i+1] - v[i]
        Udif2[i] = v[i+2] - v[i]
        Udif3[i] = v[i+3] - v[i]
        Udif4[i] = v[i+4] - v[i]
        Udif5[i] = v[i+5] - v[i]
    U_dict = {'time': trange, 'U_0': v, 'Udif1': Udif1, 'Udif2': Udif2, 'Udif3': Udif3,
              'Udif4': Udif4, 'Udif5': Udif5}
    U_df = pd.DataFrame(U_dict)
    U_df = U_df.round(5)
    U_df.to_csv(f"inlet_Us_diff/inlet_U_vp7_3.csv", index=False)
    return v, Udif1, Udif2, Udif3, Udif4, Udif5

def real_aorta(t):
    f = np.asarray([4,11,15,17, 18,20,21,23, 24,25, 26,28, 29, 28,27,26, 25,24, 22, 22,75,115,182,175,150, 128, 85,60,45, 7,4])
    return f

def fourier_series_real_aorta(T, N):
    f_sample = 2*N
    t, dt = np.linspace(0, T, f_sample, endpoint=False, retstep=True)
    y = np.fft.rfft(real_aorta(t)) /t.size
    y *=2
    return y[0].real, y[1:-1].real, -y[1:-1].imag


def p_out(t):
    """
    function to create time dependent pressure outlet boundary condition
    :param t: time vector
    :return p_list: pressure list containing corresponding pressure values
    """
    #y = [62.5,75,82,85,87,85,84,82.5,80,76,74,72,70,69,68,66,65,64,63,62.5] #values in mmHg
    y = [80, 90, 100, 110, 120, 115, 110, 112, 106, 105, 103, 100, 97, 94, 91, 87, 85, 83, 80, 80]
    x = np.linspace(0,1,20)

    p_coef = np.polyfit(x,y,10)
    p10 = np.poly1d(p_coef)
    xp10 = np.linspace(0,1,10000)

    p_list =[]
    for count, value in enumerate(t):
        p_list.append(p10(value))

    # p_list_1 = p_list.copy()
    # p_list.extend(p_list_1)
    # p_list.extend(p_list_1)

    p_conv_list = p_list.copy()
    for count, value in enumerate(p_list):
        p_conv_list[count] = p_list[count]*133.322368/1060  #convert for  OpenFoam
    print(p_conv_list)

    plt.plot(x,y, '.',xp10,p10(xp10), '-')
    plt.show()
    return p_list, p_conv_list

def plot_figure(t, p_list):
    plt.rcParams.update({'font.size': 18})
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('velocity[m/s]', color = 'b')
    U0, _,_,_,_,_ = v_sine(0.4,0.02,0.35,0.25)
    ax1.plot(t, U0, 'b')

    ax2 = ax1.twinx()
    ax2.set_ylabel('pressure [mmHg]', color = 'r')
    ax2.plot(t, p_list, 'r')
    plt.tight_layout()
    plt.title('Boundary conditions')
    plt.savefig('BC.jpg')
    plt.show()


def data_table(t, p_conv_list) -> object:
    t_list = t.tolist()
    # v_list = v_sine(0.4, 0.02,t).tolist()
    # v_dict = {'time':t_list, 'velocity':v_list}
    # v_df = pd.DataFrame(v_dict)
    # v_df = v_df.round(3)
    # v_df.to_csv('inlet_v.csv', index=False)

    p_dict = {'time':t_list, 'pressure': p_conv_list}
    p_df = pd.DataFrame(p_dict)
    p_df = p_df.round(5)
    p_df.to_csv('outlet_p.csv', index=False)

def v_in_poly():
    y = [0.08,0.21,0.62,0.53,0.37,0.22,0.02,0.05,0.07,0.08, 0.07, 0.08,  0.07, 0.08, 0.08]
    x = np.linspace(0,0.75,15)


    params, _ = optimize.curve_fit(test_func, x, y)
    plt.scatter(x,y)
    plt.plot(x, test_func(x, params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]))
    plt.show()

def init():
    line.set_data([], [])
    return line,

def animate(i):
    x = trange
    y = U0_vp5[8][:i]
    ax.set_xlim([0,1])
    ax.set_ylim([0, 0.5])
    line.set_data(x, y)
    return line,


if __name__ == "__main__":
    a0, a, b = fourier_series_real_aorta(1, 15)
    print(a0, a, b)
    v,_,_,_,_,_ = real_beat(a0,a,b)



#    data_table(t, None)
#    v_list, v_dict = v_in(t)
    #p_list, p_conv_list = p_out(t)
    #data_table(t, p_conv_list)
    #t = np.linspace(0,1,101)
    #plot_figure(t, p_list)

    # plt.plot(t, p_conv_list)
    # plt.show()
    # plot_figure(t, p_list)
    # plt.show()
    #data_table(t, p_conv_list)
    U_0 = [[] for i in range(5)]
    U0_vp5 = [[] for i in range(11)]
    U0_vp6 = [[] for i in range(11)]

    for vp in range(7):
        U0, U0_vp5, U0_vp6, t = vel_csv(vp, U_0, U0_vp5, U0_vp6)
    #
    #
    tfinal =1
    dt = 0.01
    n = int(tfinal/dt)
    trange = np.linspace(start=0, stop=tfinal, num=n+1)
    # v, _,_,_,_,_ =  real_beat(a0, a, b)
    # U0_vp5_9_3beats = np.append(U0_vp5[8], U0_vp5[8][1:])
    # U0_vp5_9_3beats = np.append(U0_vp5_9_3beats, U0_vp5[8][1:])
    #
    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # line, = ax.plot([], [], lw=2)
    # anim = animation.FuncAnimation(fig, animate, init_func = init, frames=len(trange)+1, interval=0.01, blit=True)
    # plt.show()

    markers_on = [60]
    plt.figure()
    plt.rcParams['font.size'] = '18'
    plt.plot(trange, v, '-bo', markevery=markers_on)
    plt.title('Real velocity pattern')
    plt.xlim([0, 1])
    plt.ylim([-0.01, 0.65])
    plt.xlabel('time [s]')
    plt.ylabel('velocity [m/s]')
    plt.tight_layout()
    plt.savefig('vp_real_m60.jpeg', bbox_inches='tight', pad_inches=0)
    plt.show()
    # plt.figure()
    # plt.rcParams['font.size'] = '18'
    # for i in range(5):
    #     ax = plt.plot(trange, U_0[i], label=f"vp{i}")
    # plt.legend(prop={"size":12})
    # plt.title('Velocity patterns 0 to 4')
    # plt.xlim([0, 1])
    # plt.ylim([0, 0.5])
    # plt.xlabel('time [s]')
    # plt.ylabel('velocity [m/s]')
    # plt.tight_layout()
    # plt.savefig('vp0-4.jpeg', bbox_inches='tight', pad_inches=0)
    # plt.show()
    # plt.figure()
    # plt.rcParams['font.size'] = '18'
    # for i in range(11):
    #     if i < 8:
    #         plt.plot(trange, U0_vp5[i], 'g', alpha=0.5, label=f"vp5.{i+1}")
    #     else:
    #         color = ['blue', 'm', 'r']
    #         plt.plot(trange, U0_vp5[i],color[i-8], label=f"vp5.{i+1}")
    # plt.xlim([0,1])
    # plt.ylim([0,0.5])
    # plt.xlabel('time [s]')
    # plt.ylabel('velocity [m/s]')
    # plt.title('Velocity patterns 5')
    # plt.legend(prop={"size":10}, loc = 'upper right')
    # plt.tight_layout()
    # plt.savefig('vp5_traintest.jpeg', bbox_inches='tight', pad_inches=0)
    # plt.show()
    # plt.figure()
    # plt.rcParams['font.size'] = '18'
    # plt.plot(trange, U0_vp5[1])
    # plt.xlim([0, 1])
    # plt.ylim([0, 0.5])
    # plt.title('General form velocity pattern 5')
    # plt.xlabel('time [s]')
    # plt.ylabel('velocity [m/s]')
    # plt.tight_layout()
    # plt.savefig('vp5_general.jpeg', bbox_inches='tight', pad_inches=0)
    # plt.show()
    # plt.figure()
    # plt.rcParams['font.size'] = '18'
    # c = 0
    # for i in range(11):
    #     if i == 7 or i>8:
    #         color = ['blue', 'm', 'r']
    #         plt.plot(trange, U0_vp6[i],color[c], label=f"vp6.{i+1}")
    #         c += 1
    #     else:
    #         plt.plot(trange, U0_vp6[i], 'g', alpha=0.5, label=f"vp6.{i + 1}")
    #
    # plt.title('Velocity patterns 6')
    # plt.legend(prop={"size":10}, loc='upper right')
    # plt.xlabel('time [s]')
    # plt.ylabel('velocity [m/s]')
    # plt.xlim([0, 1])
    # plt.ylim([0, 0.5])
    # plt.tight_layout()
    # plt.savefig('vp6_traintest.jpeg', bbox_inches='tight', pad_inches=0)
    # plt.show()
    # plt.figure()
    # plt.rcParams['font.size'] = '18'
    # plt.plot(trange, U0_vp6[3])
    # plt.xlim([0, 1])
    # plt.ylim([0, 0.5])
    # plt.title('General form velocity pattern 6')
    # plt.xlabel('time [s]')
    # plt.ylabel('velocity [m/s]')
    # plt.tight_layout()
    # plt.savefig('vp6_general.jpeg', bbox_inches='tight', pad_inches=0)
    # plt.show()