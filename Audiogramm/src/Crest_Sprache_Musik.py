# ---- Ton2 WiSe24/25 PA2 Gruppe 3 ----

# Crest-Faktor für Sprache und Musik


import soundfile as sf
import numpy as np

sprache, samplerate = sf.read('./spoken.mp3')
musik, samplerate2 =  sf.read('./Track 1.mp3')

def crest_factor(signal):
    peak_value = np.max(np.abs(signal))
    rms_value = np.sqrt(np.mean(signal**2))
    return peak_value / rms_value

# Berechnung des Crest-Factors
crest_sprache = crest_factor(sprache)
print(f"Crest-Factor für Sprache: {crest_sprache}")

# Berechnung des Crest-Factors
crest_musik = crest_factor(musik)
print(f"Crest-Factor für Musik: {crest_musik}")