# ---- Ton2 WiSe24/25 PA2 Gruppe 3 ----

# Crest-Faktor für verschiedene Signale

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

fs = 44100  # Abtastrate
t = np.linspace(0, 1, fs)

#Sinussignal
sinus_signal = np.sin(2 * np.pi * 100 * t)

#Rechtecksignal
square_signal = signal.square(2 * np.pi * 100 * t)

#Rauschsignal
noise_signal = np.random.normal(0, 1, len(t))

#Dreiecksignal
triangle_signal = signal.sawtooth(2 * np.pi * 100 * t, 0.5)

#Berechnung Crest-Faktor als Funktion
def crest_factor(signal):
    peak_value = np.max(np.abs(signal))  # Scheitelwert
    rms_value = np.sqrt(np.mean(signal**2))  # Effektivwert (RMS)
    return peak_value / rms_value

crest_sinus = crest_factor(sinus_signal)
crest_square = crest_factor(square_signal)
crest_noise = crest_factor(noise_signal)
crest_triangle = crest_factor(triangle_signal)

# Ausgabe der Crest-Faktoren
print(f"Crest-Factor für ein Sinussignal: {crest_sinus}")
print(f"Crest-Factor für ein Rechtecksignal: {crest_square}")
print(f"Crest-Factor für ein Rauschsignal: {crest_noise}")
print(f"Crest-Factor für ein Dreiecksignal: {crest_triangle}")



