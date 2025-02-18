import numpy as np
import sounddevice as sd

# Sampling-Rate
fs = 44100

# Anzahl der Reihen und Spalten (zeitliche Anordnung)
rows = 5  # Reihen (Häufigkeit)
cols = 4  # Spalten (Wiederholungen)

# Impulsparameter
impulsdauer = 0.001  # Sekunden
pausenzeit = 0.1  # Zeit zwischen Impulsen

# Erzeuge einen einzelnen Knack-Impuls
impuls_samples = int(impulsdauer * fs)
impuls = np.concatenate([np.ones(impuls_samples), np.zeros(int(pausenzeit * fs))])

# Rechteckige Sequenz generieren
knacks = np.tile(impuls, cols)  # In Spalten wiederholen

# Mehrere Reihen erzeugen (mit Pause zwischen den Gruppen)
pause_zwischen_reihen = np.zeros(int(0.5 * fs))  # 0.5 Sekunden Pause zwischen Reihen
output_signal = np.concatenate([knacks] * rows + [pause_zwischen_reihen])

# Abspielen
sd.play(output_signal, samplerate=fs)
sd.wait()