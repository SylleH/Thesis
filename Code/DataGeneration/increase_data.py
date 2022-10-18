import matplotlib.pyplot as plt
import numpy as np

loss = np.asarray([14, 5.3840, 5.9013, 6.2383, 3.0502])*10**-4
mae = np.asarray([0.0481,0.0288,0.0297,0.03,0.0216])
data = np.asarray([15, 28, 41, 53, 67])
#m, b = np.polyfit(data, loss, 1)
plt.rcParams.update({'font.size': 28})
fig, ax1 = plt.subplots(figsize= [10, 8])
ax1.set_xlabel('# timeseries in dataset')
ax1.set_ylabel('MSE', color='r')
ax1.plot(data, loss, 'ro', markersize=18)

ax2 = ax1.twinx()
ax2.set_ylabel('MAE' , color='b')
ax2.plot(data, mae, 'b*',markersize=18)
plt.tight_layout()
plt.savefig('res_increase_data.jpg', bbox_inches='tight', pad_inches=0)
plt.show()

def ReLU(x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = 0
    return y

def LeakyReLU(a, x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = a*x[i]
    return y

def ELU(a, x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            y[i] = x[i]
        else:
            y[i] = a*(np.exp(x[i])-1)
    return y

def SELU(l,a, x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] > 0:
            y[i] = l*x[i]
        else:
            y[i] = l*a*(np.exp(x[i])-1)
    return y

def sigmoid(x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = 1/(1+np.exp(-x[i]))
    return y

def tanh(x):
    y = np.zeros_like(x)
    for i in range(len(x)):
        y[i] = (np.exp(x[i])-np.exp(-x[i]))/(np.exp(x[i])+np.exp(-x[i]))
    return y

# x = np.linspace(start = -5, stop=5, num=1000)
# plt.figure(figsize=[6,4])
# plt.rcParams.update({'font.size': 18})
# plt.plot(x, ReLU(x), label='ReLU')
# plt.plot(x,LeakyReLU(0.1,x), label='LeakyReLU')
# plt.plot(x, ELU(1,x), 'g--', label='ELU')
# plt.plot(x, SELU(1,1.5,x),'k:', label='SELU')
# plt.plot(x, sigmoid(x),'m', label='sigmoid')
# plt.plot(x, tanh(x),'r', label='tanh')
# plt.legend(prop={"size":10})
# plt.title('Common activation functions')
# plt.ylim([-2,2])
# plt.xlim([-5,5])
# plt.tight_layout()
# plt.savefig('activation_funcs.jpg', bbox_inches='tight', pad_inches=0)
# plt.show()