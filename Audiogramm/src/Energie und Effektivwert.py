# ---- Ton2 WiSe24/25 PA2 Gruppe 3 ----

# Energie und Effektivwert Sinus


import numpy as np

fs = 44100
def t(n:int): return np.linspace(0, 1, n*fs)
sinus_signal = np.sin(2 * np.pi * 100 * t(2))

def signal_energy(signal):
    return np.sum(signal**2)/fs

def rms_value(signal):
    return np.sqrt(np.mean(signal**2))

# Berechnung der Energie und des Effektivwerts
energie_sinus = signal_energy(sinus_signal)
rms_sinus = rms_value(sinus_signal)
print(f"Energie des Sinussignals: {energie_sinus}")
print(f"Effektivwert des Sinussignals: {rms_sinus}")
