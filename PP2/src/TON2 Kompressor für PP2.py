# -*- coding: utf-8 -*-
"""

@author: E_Wilk
------------------------------------
"Tontechnik 2", Winter 2024
Prof. Dr. Eva Wilk

Kompressor und Limiter

Vorlage für Praxisproblem 2

ACHTUNG: es müssen noch geeignete Parameter im Code eingetragen werden,
   bei "!!! WERT EINGEBEN"
------------------------------------
Quellen: 
    Zölzer, DAFX, S. 109 ff
    Mathworks, https://de.mathworks.com/help/audio/ref/limiter-system-object.html 
-------
Parameter:
  Threshold, L_Thresh -50 .. 0, in dB. 
  Attack-Time, tAT = 0.02 .. 10 ms nach Zoelzer
  ReleaseTime, tRT = 1 .. 5000 ms nach Zoelzer
  
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.io.wavfile import write
from scipy.io.wavfile import read



## Zentrale Schalter
wahl_dynamik = 1  # wenn wahl_dynamik = 1: dynamische Kennlinie, wenn 0: statische Kennlinie
wahl_funktion = 'Komp' # Wahl:'Lim:' Limiter, 'Komp': Kompressor

x_ref = pow(2,15)-1  #Referenzwert für Pegel

####################################
# Einlesen Testsignal
Fs,x= read("test.wav")
anz_werte = x.size
dauer_s = anz_werte/Fs
deltat = 1./(Fs)  #Zeit-Schritt, float
t = np.arange(x.size)/float(Fs)

#umwandeln in float und normieren
x = np.array( x, dtype=np.float64)
x_norm = x/x_ref   # Normierung, falls erforderlich/gewünscht


### Kompressor-Parameter festlegen und Umrechnen in Abtastwerte
tAT = 0.01# Attack-Time, !!! WERT EINGEBEN
tRT = 0.9 # Release-Time, !!! WERT EINGEBEN

tAT_i = tAT*Fs
tRT_i = tRT*Fs

# smoothing detector filter:
faktor = np.log10(9)
a_R = np.e **(-faktor/(tAT_i))
a_T = np.e **(-faktor/(tRT_i))


## x_ref = np.abs(np.max(x))  #Referenzwert für Pegel

L_thresh = -6   #Threshold, in dB, !!! WERT EINGEBEN
u_thresh = 10**(L_thresh/20)*x_ref

R = 20 # Ratio, !!! WERT EINGEBEN
if R == 0: R = 0.1

L_M = 0   # Make up-Gain in dB  !!! WERT EINGEBEN

##################
## Kompression:
PegelMin = -95   # Pegelgrenze nach unten in dB

# Eingangssingal als Pegel:
Lx = np.zeros(anz_werte)      
Lx[1:] = 20*np.log10(np.abs(x[1:])/x_ref)    
Lx[0] = Lx[1]             # damit nicht log(0)

# Begrenzung des minimalen Pegels (mathematisch erforderlich)
for i in range(anz_werte):
    if Lx[i] < PegelMin:
        Lx[i] = PegelMin

# Vorbereitung der arrays:
Lx_c = np.zeros(anz_werte)      # Pegel(x) nach statischer Kompressor-Kennlinie
Lg_c = np.zeros(anz_werte)      # Pegel(gain) statisch (um wieviel wurde Lx gedämpft) 
Lg_s = np.zeros(anz_werte)       # Pegel(gain) dynamisch (smoothed, mit t_attack und t_release)
Lg_M = np.zeros(anz_werte)       # Pegel(gain) dynamisch (smoothed, mit t_attack und t_release) mit M
g_a = np.zeros(anz_werte)       # linearer gain dynamisch (smoothed, mit t_attack und t_release)

# Berechnung der momentanen Verstärkung/Dämpfung
## Limiter:
 
for i in range(anz_werte):
    if wahl_funktion == 'Lim':   ## Limiter
        if Lx[i] >= L_thresh:
             Lx_c[i] = L_thresh
        else:
             Lx_c[i] = Lx[i]
    else:  # Kompressor
        if Lx[i] > L_thresh:
             Lx_c[i] = L_thresh + (Lx[i]-L_thresh)/R
        else:
             Lx_c[i] = Lx[i]
        
    Lg_c[i] = Lx_c[i] - Lx[i]   # Dämpfung von Lx zum Zeitpunkt i 
    
#  dynamische Kennlinie
    Lg_s[0] = 0.0 #20*np.log10(x[0]/x_ref) #!!! Startwert für dynamische Dämpfung
    if wahl_dynamik == 1:
        if i > 0:
            if Lg_c[i] > Lg_s[i-1]:     # Release
                Lg_s[i] = a_T*Lg_s[i-1]+ (1-a_T)*Lg_c[i]                
            else:                       # Attack
                Lg_s[i] = a_R*Lg_s[i-1]+ (1-a_R)*Lg_c[i]
    else:
        Lg_s[i] = Lx_c[i]
 
# Anwenden der momentanen Verstärkung/Dämpfung
if wahl_dynamik == 1:
   Lg_M = Lg_s + L_M        

   g_a = 10**(Lg_M/20)   #lineare Verstärkung, zeitabhängig   
   y_a = x * g_a             # Ausgangssignal; hier ist das Vorzeichen im x vorhanden
else:
   g_mu = 10**(L_M/20)     # verstärkung ergibt sich aus makeup-gain
   y_a = 10**(Lx_c/20)*x_ref * g_mu     # y ist geclippter Eingang  

   for i in range (anz_werte):  # Vorzeichen ist verloren durch log, daher hinzufügen
        if x[i] < 0:
            y_a[i] = -y_a[i]

y_a = y_a/x_ref   # normieren, zur grafischen Darstellung

###################################
# Plots:
fig, ax = plt.subplots()   
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.5)  #Abstand zwischen Subplots

ax.plot(t,y_a)  # Plotten von y über t
ax.plot(t,g_a)  # Plotten von gain über t

# Einrichtung der Achsen:
ax.set_xlim(0, dauer_s)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('$t$ in s')
ax.set_ylabel('$y$($t$),$g$($t$) ')
ax.grid(True)
plt.show()


