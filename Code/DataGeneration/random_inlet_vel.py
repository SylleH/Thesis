import numpy as np

Umax = np.random.randint(200,401)
Umax = Umax/1000
Umin = np.random.randint(0,81)
Umin = Umin/1000
Utwo = np.random.randint(50, 201)
Utwo = Utwo/1000
tb = np.random.randint(10,31)
tb = tb/100
td = np.random.randint(1, 31)
td = td/100
tp = np.random.randint(1, 31)
tp = tp/100

print(Umax, Umin, Utwo, tb, td, tp)
#ToDo: couple with making inlet_U csv