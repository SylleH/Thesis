"""
Author: Sylle Hoogeveen
Functions to create velocity inlet and pressure outlet functions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize

delta_T = 0.0001
t = np.linspace(0,1,int(1/delta_T)) #one cycle

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

def v_poly(U_max, U_min, t):
    U_0 = np.zeros_like(t)
    for count, value in enumerate(t):
        if (value<=0.35):
            U_0[count] = 0.08 +U_max*(1-np.power(value-0.175,2)/(0.175*0.175))
        if ((value>0.35) and (value<=0.6)):
            U_0[count] = 0.08 +U_min*(-1+np.power(value-0.475,2)/(0.125*0.125))
        if (value>0.6):
            U_0[count] = 0.08
    return U_0

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

    p_list_1 = p_list.copy()
    p_list.extend(p_list_1)
    p_list.extend(p_list_1)

    p_conv_list = p_list.copy()
    for count, value in enumerate(p_list):
        p_conv_list[count] = p_list[count]*133.322368/1060  #convert for  OpenFoam
    print(p_conv_list)

    plt.plot(x,y, '.',xp10,p10(xp10), '-')
    plt.show()
    return p_list, p_conv_list

def plot_figure(t, p_list):
    plt.rcParams.update({'font.size': 12})
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('velocity[m/s]', color = 'b')
    ax1.plot(t, v_poly(0.6,0.02,t), 'b')

    ax2 = ax1.twinx()
    ax2.set_ylabel('pressure [mmHg]', color = 'r')
    ax2.plot(t, p_list, 'r')
    plt.savefig('BC.jpg')
    plt.show()


def data_table(t, p_conv_list):
    t_list = t.tolist()

    # v_df = pd.DataFrame(v_dict)
    # v_df = v_df.round(3)
    # v_df.to_csv('inlet_v.csv', index=False)

    p_dict = {'time':t_list, 'pressure': p_conv_list}
    p_df = pd.DataFrame(p_dict)
    p_df = p_df.round(4)
    p_df.to_csv('outlet_p.csv', index=False)

def v_in_sine():
    y = [0.08,0.21,0.62,0.53,0.37,0.22,0.02,0.05,0.07,0.08, 0.07, 0.08,  0.07, 0.08, 0.08]
    x = np.linspace(0,0.75,15)


    params, _ = optimize.curve_fit(test_func, x, y)
    plt.scatter(x,y)
    plt.plot(x, test_func(x, params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]))
    plt.show()



if __name__ == "__main__":
#    v_list, v_dict = v_in(t)
    p_list, p_conv_list = p_out(t)
    t = np.linspace(0,3,int(3/delta_T))
    plt.plot(t, p_list)
    plt.show()
#    plot_figure(t, p_list)
    data_table(t, p_conv_list)