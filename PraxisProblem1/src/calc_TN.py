'''
Ton2 – PP1 – Gruppe 4
Linus Ollmann
Mario Seibert
Tom Koryciak
Gabriel Christoforidis
'''

import os  # Für Dateipfade
from math import log10
from random import sample
import timeit

import numpy as np  # Für Berechnungen
import pylab as pl
import sounddevice as sd  # Für Audioausgabe
import soundfile as sf  # Zum Dateien einlesen
import matplotlib.pyplot as plt  # Für Plotting

'''Basisfunktionen'''

def amplitude_to_dbfs(amplitude):
    return 20 * np.log10(np.clip(amplitude, 1e-10, 100)) #Referenzwert 1 da Datein auf Werte [-1, 1] normiert (weggelassen)


def dbfs_to_amplitude(dbfs):
    return 10 ** (dbfs / 20)


def calc_mean_energy_per_sec(file, fs=44100):
    return np.sum(file ** 2) / fs


def calc_energy_in_interval(file, start_time=0, end_time=None, fs=44100):
    if end_time is None or end_time > (len(file) / fs):
        end_time = len(file) / fs

    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    energy = np.sum(file[start_sample:end_sample] ** 2)

    return energy


# Berechnet Scheitelfaktor, RMS und Crest-Faktor
def calculate_rms(waveform):
    return np.sqrt(np.mean(np.square(waveform)))


def corr_factor(signal1, signal2, fs=44100):
    return (np.sum(signal1 * signal2) / fs) / np.sqrt(calc_energy_in_interval(signal1) * calc_energy_in_interval(signal2))



def load_wav(file_path):
    file = os.path.join(os.path.dirname(__file__), file_path)
    data, sample_rate = sf.read(file)  # normiert Datei auf Wertebereich [-1, 1]
    left_channel, right_channel = split_channels(data)

    return left_channel, sample_rate


def split_channels(data):
    if len(data.shape) == 1:
        return data, data  # Mono to both channels
    return data[:, 0], data[:, 1]


def play_audio(audio_data, sample_rate):
    sd.play(audio_data, sample_rate)
    sd.wait()

def remove_zeros(arr):
    return arr[arr!=0]


'''Unsere neuen Funktionen'''
def calc_c50(file):
    energy_to_50ms = calc_energy_in_interval(file, 0, 0.05, sample_rate)
    energy_after = calc_energy_in_interval(file, 0.05, fs=sample_rate)

    #print(f"energy bis 50ms: {energy_to_50ms:.2f} | energy danach: {energy_after:.2f}")

    c50 = 10 * np.log10(energy_to_50ms / energy_after)
    return c50

def integrate_energy(file):
    return np.flip(np.flip(file**2).cumsum())
    #diese Funktion ist deutlich schneller als die untenstehende
    """integrated_energy = []
    length = len(file)
    for i in range(0, length):
        energy = np.sum(file[i:length]**2)
        integrated_energy.append(energy)
    return integrated_energy"""

def calc_TN(file_data,sample_rate = 44100):
    file_data = remove_zeros(file_data)
    duration = len(file_data) / sample_rate

    times = np.arange(0, duration, 1/sample_rate)
    energies = integrate_energy(file_data)  # Energie über Zeit integrieren

    full_energy = np.sum(file_data ** 2)
    energies_log = [10*log10(e/full_energy) for e in energies]  # Logarithmieren und in dB umrechnen

    find_closest_index = lambda array, target: np.argmin(np.abs(np.array(array) - target))

    t_5, e_5 = times[find_closest_index(energies_log, -5)], energies_log[find_closest_index(energies_log, -5)]
    t_15, e_15 = times[find_closest_index(energies_log, -15)], energies_log[find_closest_index(energies_log, -15)]
    t_25, e_25 = times[find_closest_index(energies_log, -25)], energies_log[find_closest_index(energies_log, -25)]
    t_35, e_35 = times[find_closest_index(energies_log, -35)], energies_log[find_closest_index(energies_log, -35)]

    print(f"T_10: {abs(t_5-t_15)*6:.2f}s | T_20: {abs(t_5-t_25)*3:.2f}s | T_30: {abs(t_5-t_35)*2:.2f}s")

    plt.figure(figsize=(10, 6))
    plt.plot(times, energies_log)
    plt.plot([t_5, t_25], [e_5, e_25], '--')
    plt.plot([t_5, t_35], [e_5, e_35], '--')
    plt.plot([t_15], [e_15], 'ro')
    plt.plot([t_25],[e_25], 'ro')
    plt.plot([t_35], [e_35], 'ro')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (dB)')
    plt.title('Energy over Time')
    plt.legend(['Energy', 'T_10', 'T_20', 'T_30'])
    plt.show()


'''Ablauf'''
#Beispiel
data, sample_rate = load_wav('Datei A_WS24.wav')
calc_TN(data)
