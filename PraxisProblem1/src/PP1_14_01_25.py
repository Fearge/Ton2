'''
Ton2 – PP1 – Gruppe 4
Linus Ollmann
Mario Seibert
Tom Koryciak
Gabriel Christoforidis
'''
#H

import os  # Für Dateipfade
from math import log10
from random import sample
import timeit

import numpy as np  # Für Berechnungen
import scipy as sc
import sounddevice as sd  # Für Audioausgabe
import soundfile as sf  # Zum Dateien einlesen
import matplotlib.pyplot as plt  # Für Plotting
import scipy.io.wavfile as wavfile
from matplotlib.ticker import ScalarFormatter
from scipy import signal
import librosa
from scipy.linalg import hilbert
from scipy.stats import pearsonr

'''Basisfunktionen'''


def amplitude_to_dbfs(amplitude, bottom=1e-10, top=1):
    return 20 * np.log10(
        np.clip(amplitude, bottom, top))  #Referenzwert 1 da Datein auf Werte [-1, 1] normiert (weggelassen)

#amplitude_response = 20 * np.log10(abs(response) + 1e-8)  # Amplituden in dB

def dbfs_to_amplitude(dbfs):
    return 10 ** (dbfs / 20)


def calc_mean_energy_per_sec(file, fs=44100):
    return np.sum(file ** 2) / fs


def calc_energy_in_interval(file, start_time=0.0, end_time=None, fs=44100):
    if end_time is None or end_time > (len(file) / fs):
        end_time = len(file) / fs

    start_sample = int(start_time * fs)
    end_sample = int(end_time * fs)

    energy = np.sum(file[start_sample:end_sample] ** 2)

    return energy


# Berechnet Scheitelfaktor, RMS und Crest-Faktor
def calculate_rms(waveform):
    return np.sqrt(np.mean(np.square(waveform)))

def load_wav(file_path):
    file = os.path.join(os.path.dirname(__file__), file_path)
    data, sample_rate = sf.read(file)  # normiert Datei auf Wertebereich [-1, 1]
    #sample_rate, data = wavfile.read(file)
    if len(data.shape) == 1:
        return data, sample_rate
    """left_channel, right_channel = data[:, 0], data[:, 1]
    mono = (left_channel + right_channel) / 2"""

    return data, sample_rate


def split_channels(data):
    if len(data.shape) == 1:
        return data, data  # Mono to both channels
    return data[:, 0], data[:, 1]


def play_audio(audio_data, sample_rate):
    sd.play(audio_data, sample_rate)
    sd.wait()


def remove_zeros(arr):
    return arr[arr != 0]


def cut_to_start(audio_signal):
    # Konvertiere das Signal in ein numpy-Array, falls es noch keins ist
    audio_signal = np.array(audio_signal)

    # Finde den Index des ersten Nicht-Null-Werts
    non_zero_index = np.where(audio_signal != 0)[0][0]

    # Schneide das Array ab diesem Index zu
    trimmed_signal = audio_signal[non_zero_index:]
    #print(f"[!] {len(audio_signal[0:non_zero_index])} Nullen wurden entfernt")

    return trimmed_signal


def prepend_zeros(audio_signal, num_zeros):
    # Erzeuge ein Array mit Nullen
    zeros = np.zeros(num_zeros, dtype=audio_signal.dtype)

    # Kombiniere die Nullen mit dem Originalsignal
    new_signal = np.concatenate((zeros, audio_signal))
    print(f"[!] {num_zeros} Nullen hinzugefügt")

    return new_signal

'''Unsere neuen Funktionen'''


def calc_c50(file, sample_rate=44100):
    energy_to_50ms = calc_energy_in_interval(file, 0, 0.05, sample_rate)
    energy_after = calc_energy_in_interval(file, 0.05, fs=sample_rate)

    c50 = 10 * np.log10(energy_to_50ms / energy_after)
    return c50


def calc_c80(file, sample_rate=44100):
    energy_to_80ms = calc_energy_in_interval(file, 0, 0.08, sample_rate)
    energy_after = calc_energy_in_interval(file, 0.08, fs=sample_rate)

    c80 = 10 * np.log10(energy_to_80ms / energy_after)
    return c80


#Funktion der Faltung
def foldThis(originalSounds, roomSounds):
    rate1, x = wavfile.read(originalSounds)  # Einlesen der Soundfiles
    rate2, h = wavfile.read(roomSounds)

    assert (rate1 == rate2)  # Sicherstellen gleicher Abtastraten
    # Sonst Assert-Errror

    # Aus Stereo Mono machen (Mittelwert beider Kanäle)
    x = x.mean(axis=1) if len(x.shape) > 1 else x
    h = h.mean(axis=1) if len(h.shape) > 1 else x

    # (schnelle) Faltung
    y = signal.fftconvolve(x, h)

    # Skalierung auf +-32765 und umwandeln in Integer
    # -> da 16 Bit
    # verhindert Clipping
    y /= max(abs(y))
    y *= (2 ** 15 - 1)
    y = y.astype('int16')

    # Ergebnis abspeichern
    # wavfile.write('y.wav', rate1, y) #statt abzuspeichern, direkt abspielen
    return y, rate1


def backwards_integration(file):
    return np.flip(np.flip(file ** 2).cumsum())
    #diese Funktion ist deutlich schneller als die untenstehende
    """integrated_energy = []
    length = len(file)
    for i in range(0, length):
        energy = np.sum(file[i:length]**2)
        integrated_energy.append(energy)
    return integrated_energy"""


def calc_TN(file_data, sample_rate=44100):
    file_data = remove_zeros(file_data)
    duration = len(file_data) / sample_rate

    times = np.arange(0, duration, 1 / sample_rate)
    energies = backwards_integration(file_data)  # Energie über Zeit integrieren

    full_energy = np.sum(file_data ** 2)
    energies_db = [10 * log10(e / full_energy) for e in energies]  # Logarithmieren und in dB umrechnen

    find_closest_index = lambda list, target: np.argmin(np.abs(np.array(list) - target)) # Funktion zum Finden des nächsten Wertes an einem Ziel

    t_5, e_5 = times[find_closest_index(energies_db, -5)], energies_db[find_closest_index(energies_db, -5)]
    t_15, e_15 = times[find_closest_index(energies_db, -15)], energies_db[find_closest_index(energies_db, -15)]
    t_25, e_25 = times[find_closest_index(energies_db, -25)], energies_db[find_closest_index(energies_db, -25)]
    t_35, e_35 = times[find_closest_index(energies_db, -35)], energies_db[find_closest_index(energies_db, -35)]

    print(f"T_10: {abs(t_5 - t_15) * 6:.2f}s | T_20: {abs(t_5 - t_25) * 3:.2f}s | T_30: {abs(t_5 - t_35) * 2:.2f}s")


    plt.plot(times, energies_db)
    """plt.plot([t_5, t_15], [e_5, e_15], '--')
    plt.plot([t_5, t_25], [e_5, e_25], '--')
    plt.plot([t_5, t_35], [e_5, e_35], '--')
    plt.plot([t_15], [e_15], 'ro')
    plt.plot([t_25], [e_25], 'ro')
    plt.plot([t_35], [e_35], 'ro')"""



    # Plot interpolated lines
    rt10_line = np.interp(times[find_closest_index(times,t_5):find_closest_index(times,t_15)], [t_5, t_15], [-5, -15])
    rt20_line = np.interp(times[find_closest_index(times,t_5):find_closest_index(times,t_25)], [t_5, t_25], [-5, -25])
    rt30_line = np.interp(times[find_closest_index(times,t_5):find_closest_index(times,t_35)], [t_5, t_35], [-5, -35])
    plt.plot(times[find_closest_index(times,t_5):find_closest_index(times,t_15)], rt10_line, label='Interpolated RT10')
    plt.plot(times[find_closest_index(times,t_5):find_closest_index(times,t_25)], rt20_line, label='Interpolated RT20')
    plt.plot(times[find_closest_index(times,t_5):find_closest_index(times,t_35)], rt30_line, label='Interpolated RT30')

    corr_rt10, _ = pearsonr(rt10_line, energies_db[find_closest_index(times,t_5):find_closest_index(times,t_15)])
    corr_rt20, _ = pearsonr(rt20_line, energies_db[find_closest_index(times,t_5):find_closest_index(times,t_25)])
    corr_rt30, _ = pearsonr(rt30_line, energies_db[find_closest_index(times,t_5):find_closest_index(times,t_35)])

    print(f"Correlation with RT10: {corr_rt10 * 100:.2f} %")
    print(f"Correlation with RT20: {corr_rt20 * 100:.2f} %")
    print(f"Correlation with RT30: {corr_rt30 * 100:.2f} %")

    plt.xlabel('Time (s)')
    plt.ylabel('Energy (dB)')
    plt.title('Energy over Time')
    plt.legend(['Energy', 'T_10', 'T_20', 'T_30'])
    plt.show()

'''Aufgabe: Amplitudenfrequenzgang und Spektrogramm'''

def plot_spectrogram_and_frequency_response(file, sample_rate):
    # Berechnung des Frequenzgangs
    # Optimierung: Reduzierung von worN auf 4096 für schnellere Berechnung (Nachteil: geringere Frequenzauflösung)
    #freqs, response = signal.freqz(file, worN=4096, fs=sample_rate)
    """t = np.linspace(0, 1.0, int(sample_rate * 1.0), endpoint=False)
    file = np.sin(2 * np.pi * 1000 * t)"""
    ft = np.fft.rfft(file)
    freqs = np.linspace(0, sample_rate/2, len(ft))
    mg_db = 20 * np.log10(abs(ft))

    #Ampl.-Frequenzgang erzeugen
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, mg_db)
    plt.title("Amplitudenfrequenzgang")
    plt.xlabel("Frequenz (Hz)")
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.xlim(20, 22050)  # Set x-axis range from 20 Hz to 20 kHz
    plt.ylabel("Amplitude (dB)")
    plt.grid()
    plt.subplot(2, 1, 2)

    # Optimierung: Reduzierung von n_fft auf 1024 und hop_length auf 256 für schnellere Verarbeitung (Nachteil: geringere zeitliche und frequenzielle Auflösung)

    S = librosa.stft(file, n_fft=1024, hop_length=256)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(S_db, sr=sample_rate, hop_length=256, x_axis="time", y_axis="hz", cmap="magma")
    plt.colorbar(format="%.1f dB")
    plt.title("Spektrogramm")

    plt.tight_layout()
    plt.show()

def plot_RIR_logarithmic(file, sample_rate):
    time = np.linspace(0, len(file) / sample_rate, len(file))
    file = signal.hilbert(file)
    file = abs(file)
    file = amplitude_to_dbfs(file)
    plt.figure(figsize=(10, 6))
    plt.plot(time, file)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Logarithmic Room Impulse Response')
    plt.grid()
    plt.show()

'''Ablauf'''
h_data, sample_rate = load_wav("Datei_C WS24.wav")
h_data = h_data[:, 1]
print(sample_rate)

plot_RIR_logarithmic(h_data, sample_rate)

#Aufgabe Spektrogramm
plot_spectrogram_and_frequency_response(h_data, sample_rate)

#Aufgabe TN
calc_TN(h_data, sample_rate)

"""#Aufgabe Maße
print("{0:->20} Mit Stille vor Datei {0:-<20}".format(""))
h_data = prepend_zeros(h_data,
                       2200)  # Stille vor Signal Start simulieren (C50 -> 2205 samples, C80 -> 3528 samples für -inf dB bei fs = 44100)
print(f"Erste 10 Stellen:\n{h_data[:10]}")
print(f"C50 = {calc_c50(h_data):.2f}dB")
print(f"C80 = {calc_c80(h_data):.2f}dB")

print("{0:->20} Bis zum Start gekürzt {0:-<20}".format(""))
h_data = cut_to_start(h_data)"""
print(f"Erste 10 Stellen:\n{h_data[:10]}")
print(f"C50 = {calc_c50(h_data, sample_rate):.2f} dB")
print(f"C80 = {calc_c80(h_data, sample_rate):.2f} dB")

#Aufgabe Faltung
originalSounds = "Dustbin.wav"
roomSounds = "Datei A_WS24.wav"
y, rate1 = foldThis(originalSounds, roomSounds)
#play_audio(y, rate1)