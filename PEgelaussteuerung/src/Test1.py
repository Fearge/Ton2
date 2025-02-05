import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# Scheitelfaktor berechnen
def calculate_peak_rms(waveform):
    peak_value = np.max(np.abs(waveform))
    rms_value = np.sqrt(np.mean(np.square(waveform)))
    crest_factor = peak_value / rms_value
    return peak_value, rms_value, crest_factor


# Energie berechnen
def calculate_energy(waveform, duration, sample_rate=44100):
    energy = np.sum(np.square(waveform)) / sample_rate
    return energy


# Korrelation zwischen zwei Signalen
def calculate_correlation(signal1, signal2):
    # Beide Signale auf die gleiche Länge bringen durch Kürzen der längeren
    min_len = min(len(signal1), len(signal2))
    signal1 = signal1[:min_len]
    signal2 = signal2[:min_len]
    correlation = np.corrcoef(signal1, signal2)[0, 1]
    return correlation


# Audiodatei einlesen und skalieren
def load_wav(relative_file_path):
    # Konvertiere den relativen Pfad zu einem absoluten Pfad
    base_dir = os.path.dirname(__file__)  # Verzeichnis des Skripts
    file_path = os.path.join(base_dir, relative_file_path)

    # Lade die Audiodatei
    data, sample_rate = sf.read(file_path)
    file_info = sf.info(file_path)
    print(file_info)
    print('---------')

    # Wenn Stereodatei, dann beide Spuren addieren und Amplitude durch Anzahl Kanäle teilen (2)
    if len(data.shape) > 1:
        # Mehrere Kanäle: Summe der Kanäle bilden
        data = np.sum(data, axis=1) / data.shape[1]

    return sample_rate, data

def calculate_pegel(datei):
    pegel = 20 * np.log10( (max(abs(datei))) / 1 )
    return pegel

# Beispielaufruf
sample_rate, data1 = load_wav("Kick.mp3")
sample_rate, data2 = load_wav("Musik.mp3")
sample_rate, data3 = load_wav("Musik.mp3")
print(max(abs(data)))
print(calculate_pegel(data))




