# -*-coding:utf-8 -*


#Il s'agit de la simulation du modèle de Hodgkin_Huxley 
# On représente les canaux par des probabilités d'ouvertures données par la théorie(page 190)
#Resolution par une méthode rk4

import matplotlib.pyplot as plt
import numpy as np
from math import *

# Tous les alphas avec mise du potentiel au repos a 0mV a l'aide de fits experimentaux

def alh(U):
    return 0.07 * np.exp(-U/20.)

def alm(U):
    return 0.1*(25.-U)/((np.exp((25.-U)/10.)) -1.)

def aln(U) :
    return 0.01*(10.-U) / ((np.exp((10.-U)/10.))-1.)

# Tous les betas avec le meme decalage de potentiel a l'aide des memes fits

def beh(U):
    return 1./(np.exp((30.-U)/10.)+1.)

def bem(U):
    return 4.* np.exp(-U/18.)

def ben(U) :
    return 0.125 * np.exp(-U/80.)


# FONCTION PROBA OUVERTURE des canaux avec pp: proba de passer

def ouverture(al, pp, be):
    return (al * ( 1 - pp )) - (be * pp)
 



# Définition des parametres
V_reset=0
V_stop=15
m_inf= lambda v: alm(v)/(alm(v)+bem(v))
n_inf= lambda v: aln(v)/(aln(v)+ben(v))
h_inf= lambda v: alh(v)/(alh(v)+beh(v))

   # constantes du courant de fuite en mV
EL = 10.6
gL = 0.3
    
    # constantes du courant K en mV
gK = 36.
EK = -12.
    
    # constantes du courant NA en mV
gNa = 120.
ENa = 115.

  # vecteurs de stockage et conditions initiales
tps = [0.]
V=[V_reset]
m=[m_inf(V_reset)]
n = [n_inf(V_reset)]
h = [h_inf(V_reset)]
K = [0.] #courant correspo0ndant a K
Na = [0.] #courant correspondant a Na

## variables temporelles
T     = 6    # ms
dt    = 0.025 # ms
time  = np.arange(0,T+dt,dt)

    #definition du courant imposé (stimulus)
I = np.zeros(len(time))

for i, t in enumerate(time):
	I[i] = 10 # uA/cm2

    


#Fonction de courant de fuite 
def im(V,i,n,m,h):
	
	return gl*(V-El) + gk*(n[i]**4)*(V-Ek)+ gna*(m[i]**3)*h[i]*(V-Ena) 



#Equation differentielle de V

def eqV(V,t):  #I=Rm*Ie(t)/taum

	return I[t] - (gL*(V-EL) + gK*(n[t]**4)*(V-EK)+ gNa*(m[t]**3)*h[t]*(V-ENa))





# Fonction resolvant le probleme a l'aide de rk4

def rk4HH(C=1.):   # C=cm pris egal a 1 dans les simulations, Tmax en ms
 
 
    # boucle sur le temps 
    
    for t in range(1,len(time)):
	pas=dt

#Resolution de n
        k1 = ouverture ( aln(V[ t-1 ]) ,n[t-1],ben(V[ t-1 ])) # k1 = f(y , t)
 	k2 = ouverture (aln(V[ t-1 ]), n[ t-1 ]+ pas * k1 , ben(V[ t-1 ]))
 	k3 = ouverture (aln(V[ t-1 ]), n[ t-1 ]+ pas * k2 , ben(V[ t-1 ]))
 	k4 = ouverture (aln(V[ t-1 ]), n[ t-1 ]+2* pas* k3 , ben(V[ t-1 ]) )
	n.append( n[t-1 ]+ pas /3*( k1 +2* k2 +2* k3 + k4 ))

#Resolution de m
        k1 = ouverture ( alm(V[ t-1 ]) ,m[t-1],bem(V[ t-1 ])) # k1 = f(y , t)
 	k2 = ouverture (alm(V[ t-1 ]), m[ t-1 ]+ pas * k1 , bem(V[ t-1 ]))
 	k3 = ouverture (alm(V[ t-1 ]), m[ t-1 ]+ pas * k2 , bem(V[ t-1 ]))
 	k4 = ouverture (alm(V[ t-1 ]), m[ t-1 ]+2* pas* k3 , bem(V[ t-1 ]) )
	m.append( m[t-1 ]+ pas /3*( k1 +2* k2 +2* k3 + k4 ))

#Resolution de h
        k1 = ouverture ( alh(V[ t-1 ]) ,h[t-1],beh(V[ t-1 ])) # k1 = f(y , t)
 	k2 = ouverture (alh(V[ t-1 ]), h[ t-1 ]+ pas * k1 , beh(V[ t-1 ]))
 	k3 = ouverture (alh(V[ t-1 ]), h[ t-1 ]+ pas * k2 , beh(V[ t-1 ]))
 	k4 = ouverture (alh(V[ t-1 ]), h[ t-1 ]+2* pas* k3 , beh(V[ t-1 ]) )
	h.append(h[t-1 ]+ pas /3*( k1 +2* k2 +2* k3 + k4 ))

        
#Courant dans les canaux
        cK = (gK * ((n[t-1])**4.) * (V[t-1] - EK))
        cNa = (gNa * (m[t-1])**3. * (h[t-1]) * (V[t-1]-ENa))
        K.append(cK)
        Na.append(cNa)

#Resolution de l'equation du potentiel

        k1 = eqV ( V[ t-1 ] ,t-1 ) # k1 = f(y , t)
 	k2 = eqV (V [ t-1 ]+ pas * k1 , t )
 	k3 = eqV (V [ t-1 ]+ pas * k2 , t )
 	k4 = eqV (V [ t-1 ]+2* pas* k3 ,t  )
 	V.append(V [t-1 ]+ pas /3*( k1 +2* k2 +2* k3 + k4 ))# on a deja divise h par 2

       
        
        tps.append(tps[t-1] + dt)

    return tps,V,K,Na,n,m,h 





# PLOT
tt, Vm, lK, lNa, n, m, h = rk4HH()

plt.figure(1)
plt.plot(tt, Vm,label='potentiel d action du neurone')
plt.legend()

plt.figure(2)
plt.subplot(311)
plt.plot(tt,n,label='proba n')
plt.legend()
plt.subplot(312)
plt.plot(tt,h,'r',label='proba h')
plt.legend()
plt.subplot(313)
plt.plot(tt,m,'g',label='proba m')
plt.legend()
plt.show()
