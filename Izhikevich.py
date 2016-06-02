# -*-coding:utf-8 -*

#   Programme orienté objet du modele de neurone Izhikevich 

import numpy as np
from math import *
from matplotlib.pyplot import *

class IzhNeurone:
  """Classe representant le modèle du neurone. L'intérêt de cette classe est l'initialisation 
  et de garder une trace des paramètres dans la simulation"""

  def __init__(self, label, a, b, c, d, v0, u0=None):
    """Definition des parametres du neurone selon l'équation du modele """

    self.label = label # Nom du neurone

#Parametres equations
    self.a = a
    self.b = b
    self.c = c
    self.d = d

#Condiotns initiales
    self.v = v0 
    self.u = u0 if u0 is not None else b*v0 


class IzhSimu:
  """ Cette classe va permettre d'effectuer toutes les différentes simulations du neurone
      Elle garde en trace le neurone durant la simulation, le temps total et le pas de temps, 
      le stimulus en entrée ainsi qu'une méthode d'intégration"""

  def __init__(self, n, T, dt=0.25):
    """ Définition de tous les paramètres globaux de la simulation """
    self.neuron = n
    self.dt     = dt
    self.t      = t = np.arange(0, T+dt, dt)
    self.stim   = np.zeros(len(t))
    self.x      = 5 #param constant dans l'equation
    self.y      = 140 #param constant dans l'eqaution
    self.du     = lambda a, b, v, u: a*(b*v - u)

  def integrate(self, n=None):
    """ Méthode d'intégration numérique du modèle et de son équation """

    if n is None: n = self.neuron
    trace = np.zeros((2,len(self.t))) # tableau a 2 dimensions
    for i, j in enumerate(self.stim): # Utilisation de la méthode d'Euler au premier ordre
      n.v += self.dt * (0.04*n.v**2 + self.x*n.v + self.y - n.u + self.stim[i])
      n.u += self.dt * self.du(n.a, n.b, n.v, n.u)
      if n.v > 30:
        trace[0,i] = 30 #Remise a 30 pour garder la même hauteur de pic
        n.v        = n.c
        n.u       += n.d
      else:
        trace[0,i] = n.v
        trace[1,i] = n.u
    return trace # On renvoie v et u 


#A present voici les diffentes classes de simulations a aprtir des donnees de Izhikevich : http://www.izhikevich.org/human_brain_simulation/Blue_Brain.htm

sims = []

## (A) tonic spiking
n = IzhNeurone("(A) tonic spiking", a=0.02, b=0.2, c=-65, d=6, v0=-70)
s = IzhSimu(n, T=100)
for i, t in enumerate(s.t):
  s.stim[i] = 14 if t > 10 else 0
sims.append(s)

##(B) Phasic spiking
n=IzhNeurone("(B) phasic spiking", a=0.02, b=0.25, c=-65, d=6, v0=-64)
s=IzhSimu(n,T=200)
for i, t in enumerate(s.t):
  s.stim[i] = 0.5 if t > 20 else 0
sims.append(s)

##(C) Tonic bursting
n=IzhNeurone("(C) tonic bursting", a=0.02, b=0.2, c=-50, d=2, v0=-70)
s=IzhSimu(n,T=220)
for i, t in enumerate(s.t):
  s.stim[i] = 15 if t > 22 else 0
sims.append(s)

##(D) Phasic bursting
n=IzhNeurone("(D) Phasic bursting", a=0.02, b=0.25, c=-55, d=0.05, v0=-64)
s=IzhSimu(n,T=200,dt=0.2)
for i, t in enumerate(s.t):
  s.stim[i] = 0.6 if t > 20 else 0
sims.append(s)

##(E) Mixed mode
n=IzhNeurone("(E) Mixed mode", a=0.02, b=0.2, c=-55, d=4, v0=-70)
s=IzhSimu(n,T=160)
for i, t in enumerate(s.t):
  s.stim[i] = 10 if t > 16 else 0
sims.append(s)

##(F) Spike frequency adaptation
n=IzhNeurone("(F) Spike frequency adaptation", a=0.01, b=0.2, c=-65, d=8, v0=-70)
s=IzhSimu(n,T=85)
for i, t in enumerate(s.t):
  s.stim[i] = 35 if t > 8 else 0
sims.append(s)

## (G) Class 1 exc.
n = IzhNeurone("(G) Class 1 exc.", a=0.02, b=-0.1, c=-55, d=6, v0=-60)
s = IzhSimu(n, T=300)
s.x = 4.1
s.y = 108
for i, t in enumerate(s.t):
  s.stim[i] = 0.075*(t-30) if t > 30 else 0
sims.append(s)

## (H) Class 2 exc.
n = IzhNeurone("(H) Class 2 exc.", a=0.2, b=0.26, c=-65, d=0, v0=-64)
s = IzhSimu(n, T=300)
for i, t in enumerate(s.t):
  s.stim[i] = -0.5+0.015*(t-30) if t > 30 else -0.5
sims.append(s)

## (I) Spike latency
n = IzhNeurone("(I) Spike latency", a=0.02, b=0.2, c=-65, d=6, v0=-70)
s = IzhSimu(n, T=100,dt=0.2)
for i, t in enumerate(s.t):
  s.stim[i] = 7.04 if (t > 10 and t<13)  else 0
sims.append(s)

## (J) subthreshold variation
n = IzhNeurone("(J) subthreshold variation", a=0.05, b=0.26, c=-60, d=0, v0=-62)
s = IzhSimu(n, T=200)
for i, t in enumerate(s.t):
  s.stim[i] = 2. if (t > 20 and t<25)  else 0
sims.append(s)

## (K) resonator
n = IzhNeurone("(K) resonator", a=0.1, b=0.26, c=-60, d=-1, v0=-62)
s = IzhSimu(n, T=400)
for i, t in enumerate(s.t):
  if t >40 and t<44 :     s.stim[i] = 0.65
  elif t>60 and t < 64:   s.stim[i] = 0.65
  elif t>260 and t < 264: s.stim[i] = 0.65
  elif t>300 and t<304: s.stim[i]=0.65
  else:           s.stim[i] = 0
sims.append(s)

## (L) integrator
n = IzhNeurone("(L) integrator", a=0.02, b=-0.1, c=-55, d=6, v0=-60)
s = IzhSimu(n, T=100)
s.x=4.1
s.y=108.
for i, t in enumerate(s.t):
  if t >(100./11.) and t<(100./11.)+2 :     s.stim[i] = 9.
  elif t>(100./11.)+5 and t < (100./11.)+7:   s.stim[i] = 9.
  elif t>60 and t < 62: s.stim[i] = 9.
  elif t>70 and t<72: s.stim[i]=9.
  else:           s.stim[i] = 0
sims.append(s)

## (M) rebound spike
n = IzhNeurone("(M) rebound spike", a=0.03, b=0.25, c=-60, d=4, v0=-64)
s = IzhSimu(n, T=200,dt=0.2)
for i, t in enumerate(s.t):
    s.stim[i] = -15 if (t>20 and t<25) else 0
  
sims.append(s)


## (N) rebound burst
n = IzhNeurone("(N) rebound burst", a=0.03, b=0.25, c=-52, d=0, v0=-64)
s = IzhSimu(n, T=200,dt=0.2)
for i, t in enumerate(s.t):
    s.stim[i] = -15 if (t>20 and t<25) else 0
  
sims.append(s)

## (O) threshold variability
n = IzhNeurone("(O) threshold variability", a=0.03, b=0.25, c=-60, d=4, v0=-64)
s = IzhSimu(n, T=100)
for i, t in enumerate(s.t):
  if 10 < t < 15:     s.stim[i] = 1.
  elif 80 < t < 85:   s.stim[i] = 1
  elif 70 < t < 75 : s.stim[i] = -6
  else:           s.stim[i] = 0
  
sims.append(s)

## (P) bistability
n = IzhNeurone("(P) bistability", a=0.1, b=0.26, c=-60, d=0, v0=-61)
s = IzhSimu(n, T=300)
for i, t in enumerate(s.t):
  if (300./8.) < t < (300./8. + 5):     s.stim[i] = 1.24
  elif 216 < t < 221:   s.stim[i] = 1.24
  else:           s.stim[i] = 0.24
  
sims.append(s)

## (Q) depolarizing after potential
n = IzhNeurone("(Q) DAP", a=1., b=0.2, c=-60, d=-21, v0=-70)
s = IzhSimu(n, T=50,dt=0.1)
for i, t in enumerate(s.t):
  if abs(t-10) < 1:     s.stim[i] = 20
  else:           s.stim[i] = 0
  
sims.append(s)

## (R) accomodation
n = IzhNeurone("(R) accomodation", a=0.02, b=1, c=-55, d=4, v0=-65, u0=-16)
s = IzhSimu(n, T=400, dt=0.5)
s.du = lambda a, b, v, u: a*(b*(v + 65))
for i, t in enumerate(s.t):
  if t < 200:     s.stim[i] = t/25
  elif t < 300:   s.stim[i] = 0
  elif t < 312.5: s.stim[i] = (t-300)/12.5*4
  else:           s.stim[i] = 0
sims.append(s)

## (S) inhibition-induced spiking
n = IzhNeurone("(S) IIS", a=-0.02, b=-1, c=-60, d=8, v0=-63.8)
s = IzhSimu(n, T=350,dt=0.5)
for i, t in enumerate(s.t):
  if t<50:     s.stim[i] = 80
  elif t> 250: s.stim[i]=80
  else:           s.stim[i] = 75
  
sims.append(s)

## (T) inhibition induced bursting
n = IzhNeurone("(T) IIB", a=-0.026, b=-1, c=-45, d=-2, v0=-63.8)
s = IzhSimu(n, T=350,dt=0.5)
for i, t in enumerate(s.t):
  if t<50:     s.stim[i] = 80
  elif t> 250: s.stim[i]=80
  else:           s.stim[i] = 75
  
sims.append(s)


#On trace tout a aprtir des donnes du modele

for i,s in enumerate(sims):
  res = s.integrate()
  ax  = subplot(5,4,i+1)

  ax.plot(s.t, res[0], s.t, -95 + ((s.stim - min(s.stim))/(max(s.stim) - min(s.stim)))*10)

  ax.set_xlim([0,s.t[-1]])
  ax.set_ylim([-100, 35])
  ax.set_title(s.neuron.label, size="small")
  ax.set_xticklabels([])
  ax.set_yticklabels([])
show()
