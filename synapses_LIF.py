# -*-coding:utf-8 -*

#from __future__ import division
import random
import numpy as np
import pylab as py

""" Le but est d'essayer de modeliser les synapses a l'aide de matrices, avec le modèle LIF """


## setup parameters and state variables
N    = 50                      # number of neurons
T    = 700                     # total time to simulate (msec)
dt   = 0.125                   # simulation time step (msec)
time = np.arange(0, T+dt, dt)  # time array


## LIF properties
Vm      = np.zeros([N,len(time)])  # potential (V) trace over time
tau_m   = 10                       # time constant (ms)
tau_ref = 4                        # refractory period (ms)
tau_psc = 5                        # post synaptic current filter time constant(ms)
Vth     = 1                        # spike threshold (V)
#for i in range(N):                 # random initialization (V)
#    Vm[i,0] = Vm[i,0] + random.random() 
#    


## Currents
I    = np.zeros((N,len(time)))
Iext = np.zeros(N) # externally applied stimulus
Iext[0] = 1.5 # Only on the first neuron

phase=[]
tau=[]


## Synapse weight matrix
# equally weighted ring connectivity
synapses = np.eye(N)
synapses = (np.roll(synapses, -1, 1)) # need connectivity by pairs

# randomly weighted full connectivity
#synapses = np.random.rand(N,N)*0.3


## Synapse current model, see p200 in the first book
def Isyn(t):
    '''t is an array of times since each neuron's last spike event'''
    t[np.nonzero(t < 0)] = 0
    return t*np.exp(-t/tau_psc)
last_spike = np.zeros(N) - tau_ref

## Simulate network
raster = np.zeros([N,len(time)])*np.nan #Une histoire de tracé de fonctions recup sur internet
for k in range(20):
 tau_psc=tau_psc+1
 last_spike = np.zeros(N) - tau_ref
 I    = np.zeros((N,len(time)))
 Iext = np.zeros(N) # externally applied stimulus
 Iext[0] = 1.5 # Only on the first neuron
 for i, t in enumerate(time[1:],1): # 1 initialement
    active = np.nonzero(t > last_spike + tau_ref)
    Vm[active,i] = Vm[active,i-1] + (-Vm[active,i-1] + I[active,i-1]) / tau_m * dt # Eulers method

    spiked = np.nonzero(Vm[:,i] > Vth)
    last_spike[spiked] = t
    raster[spiked,i] = spiked[0]+ 1
    I[:,i] = Iext + synapses.dot(Isyn(t - last_spike))


## plot membrane potential trace
py.plot(time, np.transpose(raster), 'b.')
py.title('Recurrent Network Example')
py.ylabel('Neurone qui spike')
py.xlabel('Time (msec)')
py.ylim([0.75,N+0.25])

py.figure(2)
py.subplot(211)
py.plot((Vm[:,0]),'r')
py.subplot(212)
py.plot(time,Vm[1,:],'b')


py.show()
