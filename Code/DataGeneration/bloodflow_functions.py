"""
Author: Sylle Hoogeveen
Functions to create velocity inlet and pressure outlet functions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

delta_T = 0.001
t = np.linspace(0,1,int(1/delta_T))

def v_in(t):
    """
    function to create time dependent velocity inlet boundary condition
    velocity pulse from t=0 to t=0.5, constant from t=0.5 to t=1
    :param t: time vector
    :return v_list: velocity list containing corresponding velocity values
    """
    y = [0.08,0.21,0.62,0.53,0.37,0.22,0.02,0.05,0.07,0.08]
    x = np.linspace(0,0.5,10)

    v_coef = np.polyfit(x,y,5)
    p6 = np.poly1d(v_coef)
    xp6 = np.linspace(0,0.5,500)

    vx_list = []
    vy_list = []
    vz_list = []

    v_list = [] #for plotting
    for count, value in enumerate(t):
        if value<0.5:
            v_list.append(p6(value))
            vy_list.append(-p6(value))
        else:
            v_list.append(0.08)
            vy_list.append(-0.08)
        vx_list.append(0)
        vz_list.append(0)

    print(v_list)
    t_list = t.tolist()
    v_dict = {'time': t_list, 'velocity_x': vx_list, 'velocity_y': vy_list, 'velocity_z': vz_list}

    plt.plot(x,y, '.',xp6,p6(xp6), '-')
    plt.show()
    return v_list, v_dict

def p_out(t):
    """
    function to create time dependent pressure outlet boundary condition
    :param t: time vector
    :return p_list: pressure list containing corresponding pressure values
    """
    y = [62.5,75,82,85,87,85,84,82.5,80,76,74,72,70,69,68,66,65,64,63,62.5] #values in mmHg
    x = np.linspace(0,1,20)

    p_coef = np.polyfit(x,y,10)
    p10 = np.poly1d(p_coef)
    xp10 = np.linspace(0,1,1000)

    p_list =[]
    for count, value in enumerate(t):
        p_list.append(p10(value))

    p_conv_list = p_list.copy()
    for count, value in enumerate(p_list):
        p_conv_list[count] = p_list[count]*133.322368/1060  #convert for  OpenFoam
    print(p_conv_list)

    plt.plot(x,y, '.',xp10,p10(xp10), '-')
    plt.show()
    return p_list, p_conv_list

def plot_figure(t, v_list, p_list):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('velocity[m/s]', color = 'c')
    ax1.plot(t, v_list, 'c')

    ax2 = ax1.twinx()
    ax2.set_ylabel('pressure [mmHg]', color = 'r')
    ax2.plot(t, p_list, 'r')
    plt.savefig('BC.jpg')
    plt.show()


def data_table(t, v_dict, p_conv_list):
    t_list = t.tolist()

    v_df = pd.DataFrame(v_dict)
    v_df = v_df.round(3)
    v_df.to_csv('inlet_v.csv', index=False)

    p_dict = {'time':t_list, 'pressure': p_conv_list}
    p_df = pd.DataFrame(p_dict)
    p_df = p_df.round(3)
    p_df.to_csv('outlet_p.csv', index=False)

if __name__ == "__main__":
    v_list, v_dict = v_in(t)
    p_list, p_conv_list = p_out(t)
    plot_figure(t, v_list,p_list)
    data_table(t, v_dict, p_conv_list)