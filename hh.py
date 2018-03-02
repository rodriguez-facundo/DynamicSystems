# by F. Rodriguez
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import RK23
'''
plt.style.use('seaborn-darkgrid')

# UNITS --> C : mF/cm^2   g : mS/cm^2     V : mV      i : uA/cm^2     t : msec

# TIME PARAMETERS
T = 40
max_step = 0.1

# INJECTED CURRENT PARAMETERS
x = [i for i in range(40)]
ext_current = np.array(x+x[::-1])
#ext_current = np.arange(0, 40, 0.5)

# INITIAL CONDITIONS
init = [ [-65, 0.05, 0.95, 0.05],   # [V, m, h, n]
         [-65, 0.95, 0.05],         # [V, h, n]
         [-65, 0.05],               # [V, n]
       ]

# CONSTANTS
C = 1.0;
g_Na = 120.0;   g_K = 36.0;     g_l = 0.3
V_Na = 50.0;    V_K = -77.0;    V_l = -54.387

# VOLTAGE GATED CHANNELS KINETIC
def alpha_m(V): return 0.1*(V+40.0)/(1.0 - np.exp(-(V+40.0) / 10.0))
def beta_m(V) : return 4.0*np.exp(-(V+65.0) / 18.0)
def m_inf(V)  : return alpha_m(V) / (alpha_m(V)+beta_m(V))
def alpha_h(V): return 0.07*np.exp(-(V+65.0) / 20.0)
def beta_h(V) : return 1.0/(1.0 + np.exp(-(V+35.0) / 10.0))
def h_inf(V)  : return alpha_h(V) / (alpha_h(V)-beta_h(V))
def alpha_n(V): return 0.01*(V+55.0)/(1.0 - np.exp(-(V+55.0) / 10.0))
def beta_n(V) : return 0.125*np.exp(-(V+65) / 80.0)
def n_inf(V)  : return alpha_n(V) / (alpha_n(V)-beta_n(V))

# MEMBRANE CURRENT
def I_Na(V,m,h): return g_Na * m**3 * h * (V - V_Na)  # Soldium
def I_K(V, n)  : return g_K * n**4 * (V - V_K)  # Potassium
def I_l(V)     : return g_l * (V - V_l)  #Leak

# ORDINARY DIFFERENTIAL EQUATIONS
def ode4(t, y):
    V, m, h, n = y
    dVdt = (I_inj(t) - I_Na(V, m, h) - I_K(V, n) - I_l(V)) / C
    dmdt = alpha_m(V)*(1.0-m) - beta_m(V)*m
    dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
    dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
    return dVdt, dmdt, dhdt, dndt

def ode3(t, y):
    V, h, n = y
    dVdt = (I_inj(t) - I_Na(V, m_inf(V), h) - I_K(V, n) - I_l(V)) / C
    dhdt = alpha_h(V)*(1.0-h) - beta_h(V)*h
    dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
    return dVdt, dhdt, dndt

def ode2(t, y):
    V, n = y
    h = 0.84438 - n
    dVdt = (I_inj(t) - I_Na(V, m_inf(V), h) - I_K(V, n) - I_l(V)) / C
    dndt = alpha_n(V)*(1.0-n) - beta_n(V)*n
    return dVdt, dndt


# find frequency from voltage
def find_frequency(Voltage, time):
    peaks = np.where(np.diff(np.sign(Voltage)))[0]
    return 1.e3 / (time[peaks[-1]]-time[peaks[-3]]) if len(peaks)>3 else 0

frequency = [list() for i in range(3)]
# SOLVE HH-MODELS
for stimulus in ext_current:
    # EXTERNAL CURRENT
    def I_inj(t): return stimulus
    
    print(init[0])
    # INTEGRATION METHOD
    X = [   RK23(ode4, t0=0.00, y0=init[0], t_bound=T, max_step=max_step),
            RK23(ode3, t0=0.00, y0=init[1], t_bound=T, max_step=max_step),
            RK23(ode2, t0=0.00, y0=init[2], t_bound=T, max_step=max_step)
            ]

    for i, x in enumerate(X):
        # restart variables
        time = [0]
        output = [x.y]

        # Solve full HH-model
        while x.status=='running':
            x.step()
            time.append(x.t)
            output.append(x.y)
        output = np.array(output)

        # next simulation continues from previews values
        init[i] = x.y
        
        # Frecuency for this imput current
        frequency[i].append(find_frequency(output[:,0], time))

# plotting
plt.title('Hogdking-Huxley models')
plt.plot(ext_current, frequency[0], label='Full')
plt.plot(ext_current, frequency[1], label='$m_{inf}=m_{(t)}$')
plt.plot(ext_current, frequency[2], label='$m_{inf}=m_{(t)} \quad and \quad h+n=cte$')
plt.xlabel('Stimulus ($u{A}/cm^2$)')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.show()






# HYPERPOLARIZATION FIRING EXAMPLE
def I_inj(t): return -4*(t>40) + 4*(t>140)

# INTEGRATION METHOD
X4 = RK23(ode4, t0=0.00, y0=init, t_bound=200, max_step=max_step)

# restart variables
time = [0];   output = [X4.y]

# iterate
while X4.status=='running':
    X4.step()
    time.append(X4.t)
    output.append(X4.y)

output = np.array(output)

plt.title("Hyperpolarization firing")
plt.plot(time, output[:,0], 'b', label="Voltage (mV)")
plt.plot(time, [I_inj(i) for i in time], 'r', label='Current $m{A}/cm^2$')
plt.legend()
plt.xlabel("Time $m{s}$")
plt.ylabel("Units")
plt.show()



# # Integrated & Fire
# tau = 1;    tauA = 1; A0 = 1; I=1
# def odeIF(t, y):
#     V, A = y
#     dVdt = - V + I - A
#     dAdt = -A + A0 * 1 if V>=1 else 0
#     return dVdt, dAdt
#
# # SOLVE
# X = RK23(odeIF, t0=0.00, y0=[-65, 1], t_bound=40, max_step=0.2)
#
# # restart variables
# time = [0]; output = [X.y]
#
# # Solve full HH-model
# while X.status=='running':
#     X.step()
#     time.append(X.t)
#     output.append(X.y)
# output = np.array(output)
#
# plt.title("Adaptive Integrated & Fire Model")
# plt.plot(time, output[:,0], 'b', label="Voltage (mV)")
# plt.ylabel("Voltage ($m{V}$)")
# plt.xlabel("Time ($m{s}$)")
# plt.show()


# # FitzHugh-Nagumo
a = 0.5
m = 0.03
I = [-0.04, 0.01, 0.055, 0.1]
#
# x = np.arange(-0.1,1.1,0.01)
# y = x * (a-x) * (x-1)
#
# plt.title("FitzHugh-Nagumo Model")
#
# plt.plot(x, y+I[0], label='I='+str(I[0]))
# plt.plot(x, y+I[1], label='I='+str(I[1]))
# plt.plot(x, y+I[2], label='I='+str(I[2]))
# plt.plot(x, y+I[3], label='I='+str(I[3]))
# plt.plot(x,m*x, 'k', label='$w= \alpha $')
# plt.legend()
# plt.xlabel("Voltage [mV]")
# plt.ylabel("w  [unknown]")
#
# plt.plot
# plt.show()


print(np.roots([-1, a+1, -(a+m), I[0]])




plt.title("FitzHugh-Nagumo Model")

plt.plot(x, y, label='I='+str(i))
plt.plot(x,m*x, 'k', label='$w= \alpha $')
plt.xlabel("Voltage [mV]")
plt.ylabel("w  [unknown]")
plt.ylim( (-0.06, 0.12) )
plt.xlim( (-0.1, 1.1) )
roots = np.roots([-1, a+1, -(a+m), i])
roots = np.array([r for r in roots if not isinstance(r, complex)])
plt.plot(roots, m*roots, 'o')
plt.legend(bbox_to_anchor=(1.1, 0.5), bbox_transform=plt.gcf().transFigure)


# FitzHugh-Nagumo
plot, i=(-0.04,0.1, 0.005), m=(0.01, 0.06, 0.005), a=(0, 1, 0.1)

def derX(x, y, a=0.5, i=0.03): return x*(a-x)*(x-1) + i - y 
def derY(x, y, m=0.03): return -y + m*x 

x = np.arange(-0.1, 1.1, 0.1)
y= np.arange(-0.06, 0.12, 0.01)

X, Y = np.meshgrid(x, y)

vX = derX(X, Y)
vY = derY(X, Y)

Q = plt.quiver(X, Y, vX, vY, units='width')
plt.show()

'''
tau = 1.8
tauA = 5.2
V0 = 0.00
A0 = 1.00
I = 1.1
Vref = 1

expo = (tauA-tau) / (tauA*tau)

def func(t): return np.exp(-t/tau) * (V0 + I*(np.exp(t/tau)-1) - A0/tau*expo*(np.exp(t*expo)-1))

y = list()
x = np.arange(0, 200, 0.01)
for i in x:
    if func(i)<Vref:
        y.append(func(i))
    else:
        break

        
plt.plot(y)
plt.show()
