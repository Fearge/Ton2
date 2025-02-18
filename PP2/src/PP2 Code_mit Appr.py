'''
Ton2 – PP2 – Gruppe 4
Mario Seibert - 2711151
'''

import os
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sounddevice as sd
import soundfile as sf
from scipy.optimize import minimize
from adjustText import adjust_text



# Wellenformen erzeugen
def generate_waveform(wave_type, frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    if wave_type == 'sine':
        waveform = np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'square':
        waveform = sg.square(2 * np.pi * frequency * t)
    elif wave_type == 'sawtooth':
        waveform = sg.sawtooth(2 * np.pi * frequency * t)
    elif wave_type == 'triangle':
        waveform = sg.sawtooth(2 * np.pi * frequency * t, 0.5)
    else:
        raise ValueError("Unsupported waveform type")

    return waveform, t

def load_wav(file_path):
    file = os.path.join(os.path.dirname(__file__), file_path)
    data, sample_rate = sf.read(file)  # normiert Datei auf Wertebereich [-1, 1]

    left_channel, right_channel = split_channels(data)

    return left_channel, sample_rate


def split_channels(data):
    if len(data.shape) == 1:
        return data, data  # Mono to both channels
    return data[:, 0], data[:, 1]


def save_audio(file_path, audio_data, sample_rate, subtype='PCM_16'):
    """
    Speichert ein Audiosignal als WAV-Datei.

    Parameters:
        file_path (str): Pfad zur Ausgabedatei.
        audio_data (array-like): Audiodaten.
        sample_rate (int): Abtastrate in Hz.
        subtype (str): Audio-Subtype (z. B. PCM_16, PCM_24, FLOAT).
    """
    sf.write(file_path, audio_data, sample_rate, subtype=subtype)


def approximate_linear(a0, N = 16):
    """
    Approximiert eine lineare Kennlinie y = a * x + b mit einer e-Funktion der Form y = c*e^(d * x)
    anhand der Methode der kleinsten Quadrate
    Parameters
    ----------
    a0 = Steigung der linearen Kennlinie
    N = Anzahl der Auflösungsbits
    dyn_range = nuerische Grenzen des Dynamikbereichs

    Returns
    -------
    a,b,ap,error Parameter und arithmetischer Fehler der Approximation
    """


    # lineare Kennlinie
    x = np.linspace(-1, 1, 2**N)
    y = a0 * x

    # e-Funktion
    def f(params):
        a, b, ap = params
        y_exp = np.exp(a * (x-ap))-b
        return np.sum((y_exp - y) ** 2)

    # Minimierungsproblem lösen
    initial_guess = [1,1,1]
    result = minimize(f, initial_guess)
    a,b,ap = result.x

    # arithmetischer Fehler
    error = np.sum((y - (np.exp(a * (x-ap))-b))) / len(y)

    # beide Funktionen um den Nullpunkt plotten
    plt.figure(figsize=(8, 6))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth=0.5)
    plt.grid(which='minor', linestyle='--', linewidth=0.3)
    plt.plot(x, y, label='Lineare Kennlinie')
    plt.plot(x, np.exp(a * (x-ap)) - b, label='Exponentielle Kennlinie')
    plt.title('Ideale und approximierte Kennlinie der Form $y = e^{b \cdot x}$')
    plt.xlabel('Eingangssignal')
    plt.ylabel(f'Ausgangssignal')
    plt.legend()
    plt.show()
    return a,b,ap, error


# System A
def process_systemA(input_signal, a, b, work_point, show_plot=True):
    """
    Prozessiert ein Eingangssignal mit der Kennlinie y1(x) = e^(a * (x - work_point)) - b und plottet die Kennlinie.

    Parameters:
        input_signal (array-like): Eingangssignal (x-Werte).
        a (float): Parameter a der Kennlinie.
        b (float): Parameter b der Kennlinie.
        work_point (float): Arbeitspunkt auf der x-Achse.
        show_plot (bool): Gibt an, ob die Kennlinie geplottet werden soll.

    Returns:
        output_signal (array-like): Ausgangssignal (y-Werte).
    """
    if show_plot==True:
        # Plot der Kennlinie
        x_values_for_curve = np.linspace(-1.2, 1.2, 500)
        y_values_for_curve = np.exp(a * (x_values_for_curve - work_point)) - b

        plt.figure(figsize=(8, 6))
        plt.plot(x_values_for_curve, y_values_for_curve, color="blue",
                 label=r'Kennlinie: $y_1(x) = e^{a \cdot (x - x_{\text{AP}})} - b$' f"\n(mit a = {a}, b = {b})")

        # Raster Strichstärken bestimmen
        plt.minorticks_on()
        plt.grid(which='major', linestyle='-', linewidth=0.5)
        plt.grid(which='minor', linestyle='--', linewidth=0.3)

        # Anzahl der Zwischenstriche festlegen
        ax = plt.gca()
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # 2 Unterteilungen zwischen Major-Ticks auf der x-Achse
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))  # 2 Unterteilungen auf der y-Achse

        # Arbeitspunkt berechnen und markieren
        work_point_y = np.exp(a * (work_point - work_point)) - b  # entspricht y1(work_point)
        plt.scatter([work_point], [work_point_y], color='red', zorder=3, label=f'Arbeitspunkt bei x = {work_point}')
        # Arbeitspunkt beschriften
        plt.text(work_point, work_point_y, f'({work_point:.1f}, {work_point_y:.1f})',
                 fontsize=10, ha='left', va='top', color='red',)

        plt.axhline(0, color="black", linewidth=1, linestyle="-")  # Nullachse
        plt.axvline(0, color="black", linewidth=1, linestyle="-")  # y-Achse

        plt.title('System A: Kennlinie mit Arbeitspunkt')
        plt.xlabel('Eingangssignal x[n]')
        plt.ylabel('Ausgangssignal y[n]')
        plt.legend()
        plt.show()

    # Verarbeitung des Eingangssignals
    output_signal = np.exp(a * (input_signal - work_point)) - b
    return output_signal


# System B - Kompressor
def process_systemB(x, Fs, typ='Komp', dynamisch=1, tAT=0.01, tRT=0.9, L_thresh=-6, R=20, L_M=0, show_plot=True):
    """
    Funktion zur Verarbeitung eines Audiosignals mit einem Kompressor oder Limiter.

    :param x: Normierte Amplitudenwerte des Audiosignals (-1 bis 1)
    :param Fs: Abtastfrequenz in Hz
    :param typ: 'Komp' für Kompressor, 'Lim' für Limiter
    :param dynamisch: 1 für dynamische Kennlinie, 0 für statische
    :param tAT: Attack-Time in Sekunden
    :param tRT: Release-Time in Sekunden
    :param L_thresh: Threshold in dB
    :param R: Ratio (Kompressionsverhältnis)
    :param L_M: Make-up-Gain in dB
    :param show_plot: Gibt an, ob das Ergebnis geplottet werden soll
    """

    anz_werte = x.size
    dauer_s = anz_werte / Fs
    t = np.arange(x.size) / float(Fs)

    # Pegelberechnung in dB
    PegelMin = -95  # Mindestpegel in dB
    Lx = np.zeros(anz_werte)
    Lx[1:] = 20 * np.log10(np.maximum(np.abs(x[1:]), 1e-10))
    Lx[0] = Lx[1]
    Lx[Lx < PegelMin] = PegelMin

    # Attack- und Release-Zeiten in Abtastwerte umrechnen
    tAT_i = tAT * Fs
    tRT_i = tRT * Fs
    faktor = np.log10(9)
    a_R = np.e ** (-faktor / tAT_i)
    a_T = np.e ** (-faktor / tRT_i)

    # Schwellenwert auf lineare Amplitude umrechnen
    u_thresh = 10 ** (L_thresh / 20)
    if R == 0:
        R = 0.1

    # Pegel Arrays
    Lx_c = np.zeros(anz_werte)
    Lg_c = np.zeros(anz_werte)
    Lg_s = np.zeros(anz_werte)
    Lg_M = np.zeros(anz_werte)
    g_a = np.zeros(anz_werte)

    # Berechnung der Pegelreduktion
    for i in range(anz_werte):
        if typ == 'Lim':
            Lx_c[i] = L_thresh if Lx[i] >= L_thresh else Lx[i]
        else:
            Lx_c[i] = L_thresh + (Lx[i] - L_thresh) / R if Lx[i] > L_thresh else Lx[i]

        Lg_c[i] = Lx_c[i] - Lx[i]

    # Dynamische Pegelsteuerung
    Lg_s[0] = 0.0
    for i in range(1, anz_werte):
        if Lg_c[i] > Lg_s[i - 1]:
            Lg_s[i] = a_T * Lg_s[i - 1] + (1 - a_T) * Lg_c[i]
        else:
            Lg_s[i] = a_R * Lg_s[i - 1] + (1 - a_R) * Lg_c[i]

    # Anwenden des Gains
    if dynamisch == 1:
        Lg_M = Lg_s + L_M
        g_a = 10 ** (Lg_M / 20)
        y_a = x * g_a
    else:
        g_mu = 10 ** (L_M / 20)
        y_a = 10 ** (Lx_c / 20) * g_mu
        y_a[x < 0] *= -1

    if show_plot==True:
        # Plotten der Ergebnisse
        fig, ax = plt.subplots()
        plt.subplots_adjust(hspace=0.5)
        ax.plot(t, y_a, label='Ausgangssignal')
        ax.plot(t, g_a, label='Gain')
        ax.set_xlim(0, dauer_s)
        #ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel('$t$ in s')
        ax.set_ylabel('$y$($t$),$g$($t$)')
        ax.grid(True)
        ax.legend()
        plt.show()

    return y_a, Fs

# Ein- und Ausgangssignal in einem Plot darstellen
def plot_waveforms(time, input_signal, output_signal, max_time=None):
    """
    Plottet die Waveforms des Eingangs- und Ausgangssignals in einem einzigen Diagramm.

    Parameters:
        time (array-like): Zeitvektor.
        input_signal (array-like): Eingangssignal.
        output_signal (array-like): Ausgangssignal.
        max_time (float, optional): Maximale Zeit in Sekunden, die geplottet werden soll.
    """
    if max_time is not None:
        # Begrenzen der Daten auf die gewünschte maximale Zeit
        mask = time <= max_time
        time = time[mask]
        input_signal = input_signal[mask]
        output_signal = output_signal[mask]

    # DC-Offset für die Darstellung entfernen
    output_signal_centered = output_signal - np.mean(output_signal)

    # Verstärkung berechnen
    gain, gain_db = calculate_gain(input_signal, output_signal)

    plt.figure(figsize=(12, 6))
    plt.plot(time, input_signal, label='Eingangssignal', color='blue', alpha=0.7)
    plt.plot(time, output_signal_centered, label='Ausgangssignal (DC Offset bereinigt)', color='red', alpha=0.7)

    # Verstärkungswerte als Text ins Diagramm schreiben
    text_x = time[int(len(time) * 0.02)]  # Position für das Label (2% der Zeitachse)
    text_y = max(np.max(output_signal_centered),
                 np.max(input_signal)) * 0.9  # Position für das Label (90% der max. Amplitude)

    plt.text(text_x, text_y, f'Gain: {gain:.2f}\nGain (dB): {gain_db:.1f} dB', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    plt.title('Ein- und Ausgangssignal im Zeitbereich')
    plt.axhline(0, color="black", linewidth=1, linestyle="-")  # Nullachse
    plt.xlabel('Zeit [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()

"""Klirrfaktor"""
def plot_spectrum_with_peaks(fft_magnitude, sample_rate, peaks, value, title):
    """
    Plottet das Spektrum mit annotierten Peaks.

    Parameter:
        fft_magnitude (np.ndarray): Magnitude der FFT.
        sample_rate (int): Abtastrate in Hz.
        peaks (np.ndarray): Indizes der Peaks.
        value (float): Wert zur Anzeige im Titel.
        title (str): Titel des Plots.
    """

    freqs = np.linspace(0, sample_rate//2, len(fft_magnitude))

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_magnitude, label='Spectrum')
    plt.plot(freqs[peaks], fft_magnitude[peaks], 'ro', label='Peaks')  # Mark the peaks


    for peak in peaks:
        plt.annotate(f'{fft_magnitude[peak]:.2f}\n{freqs[peak]:.1f} Hz',
                 (freqs[peak], fft_magnitude[peak]),
                 textcoords="offset points", xytext=(0, 10),
                 ha='center')  # Annotate the peaks with magnitude and frequency

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Spectrum with Peaks ({title}: {value:.2f}%)')
    plt.legend()
    plt.grid()
    plt.show()

def get_peaks(signal, distance, height_threshold):
    """
    Ermittelt die Peaks des Spektrums eines Signals.

    Parameter:
        signal (np.ndarray): Eingangssignal.
        distance (int): Mindestabstand zwischen Peaks.
        height_threshold (float): Mindesthöhe der Peaks.

    Rückgabe:
        tuple: Indizes der Peaks und die Magnitude der FFT.
    """
    # Remove DC component
    signal = signal - np.mean(signal)

    # returns the magnitude of the peaks of the spectrum + the spectrum
    fft_magnitude = np.abs(np.fft.rfft(signal))
    fft_magnitude = fft_magnitude / np.max(fft_magnitude)

    # Identify the peaks in the spectrum
    peak_freqs, _ = sg.find_peaks(fft_magnitude,height=height_threshold,distance=distance)

    return peak_freqs, fft_magnitude

def calc_thd(signal, sample_rate,distance = 100, height_threshold = 0.01, threshold = 0.01):
    """
        Berechnet die Gesamtklirrfaktor (THD) eines Signals.

        Parameter:
            signal (np.ndarray): Eingangssignal.
            sample_rate (int): Abtastrate in Hz.
            distance (int): Mindestabstand zwischen Peaks.
            height_threshold (float): Mindesthöhe der Peaks.
            threshold (float): Schwellenwert für die Berücksichtigung von Harmonischen.

        Rückgabe:
            float: THD-Wert.
        """
    peaks, mag = get_peaks(signal, distance, height_threshold)
    print(peaks)
    peaks_mag = mag[peaks]
    fundamental = peaks_mag[0]

    harmonics = peaks_mag[1:]
    harmonics = harmonics[harmonics > threshold]

    thd = np.sqrt(np.sum(harmonics**2)/fundamental**2)*100
    plot_spectrum_with_peaks(mag, sample_rate,peaks, thd, "THD")
    return thd

def calc_klirr(signal, sample_rate, distance = 100, height_threshold = 0.01, threshold = 0.01):
    """
        Berechnet den Klirrfaktor eines Signals.

        Parameter:
            signal (np.ndarray): Eingangssignal.
            sample_rate (int): Abtastrate in Hz.
            distance (int): Mindestabstand zwischen Peaks.
            height_threshold (float): Mindesthöhe der Peaks.
            threshold (float): Schwellenwert für die Berücksichtigung von Harmonischen.

        Rückgabe:
            float: Klirrfaktor-Wert.
        """
    peaks, mag = get_peaks(signal, distance, height_threshold)
    print(peaks)
    peaks_mag = mag[peaks]

    harmonics = peaks_mag[1:]
    harmonics = harmonics[harmonics > threshold]

    all_peaks = peaks_mag[peaks_mag > threshold]
    klirr = np.sqrt(np.sum(harmonics ** 2) / np.sum(all_peaks ** 2)) * 100

    plot_spectrum_with_peaks(mag, sample_rate, peaks, klirr, "Klirrfaktor")
    return klirr


#Verstärkungsberechnung
def calculate_gain(input_signal, output_signal):
    """
    Berechnet die Systemverstärkung als das Verhältnis der RMS-Werte von Ausgangs- und Eingangssignal.

    Parameters:
        input_signal (array-like): Eingangssignal.
        output_signal (array-like): Ausgangssignal.

    Returns:
        float: Verstärkungsfaktor des Systems.
    """
    # RMS-Wert des Eingangssignals berechnen
    rms_input = np.sqrt(np.mean(np.square(input_signal)))

    # RMS-Wert des Ausgangssignals berechnen mit DC-Offset entfernt
    output_signal_centered = output_signal - np.mean(output_signal)
    rms_output = np.sqrt(np.mean(np.square(output_signal_centered)))

    # Vermeidung einer Division durch Null
    if rms_input == 0:
        raise ValueError("Das Eingangssignal hat einen RMS-Wert von 0, Verstärkung kann nicht berechnet werden.")

    # Verstärkungsfaktor berechnen
    gain = rms_output / rms_input
    gaindb = 20 * np.log10(gain)

    print("Die Verstärkung beträgt:", f"{gain:.1f}", "\nDie Verstärkung in dB beträgt:", f"{gaindb:.1f}", "dB.")

    return gain, gaindb


def play_audio(audio_data, sample_rate):
    sd.play(audio_data, sample_rate)
    sd.wait()



# Hauptteil
if __name__ == "__main__":
    #approximate_linear(3)
    print("{0:–>23} Laden der Eingangssignale {0:–<23}".format(""))
    print("<Siehe Code>")
    # Testsinus
    sr_waveform = 44100
    input_waveform, t_waveform = generate_waveform('sine', 50, 0.1, sr_waveform)

    # Testdatei
    input_file, sr_file = load_wav("Sprechen_1.wav")
    t_file = np.linspace(0, len(input_file) / sr_file, num=len(input_file), endpoint=False)



    print("{0:–>23} I: Parameterbestimmung System A {0:–<23}".format(""))
    a, b, AP, err = approximate_linear(1.1) # Bestimmung Parameter mittels Approximation, gewünschte Verstärkung Funktion übergeben
    print(f"System A mit: a = {a :.2f}, b = {b :.2f}, AP = {AP :.2f}")
    print(f"Arithmetischer Fehler der Approximation: {err * 100:.2f}")


    print("{0:–>22} II-System A: Verstärkung und Klirrfaktor {0:–<21}".format(""))
    print("<Siehe Plots>")
    #Klirrfaktor und THD
    a= 3 # Wenn Kennlinienparameter selbst gesetzt werden sollen
    b= 0
    AP = -1
    output_waveformA = process_systemA(input_waveform, a, b, AP, show_plot=False)
    plot_waveforms(t_waveform, input_waveform, output_waveformA, max_time=0.02)
    output_waveformA = output_waveformA/np.max(np.abs(output_waveformA))
    plot_waveforms(t_waveform, input_waveform, output_waveformA, max_time=0.02)
    klirr = calc_klirr(output_waveformA, sr_waveform, 100, height_threshold=0.01)
    print(f"Klirrfaktor: {klirr:.2f}%")


    print("{0:–>23} II-System B: Verstärkung und Klirrfaktor {0:–<23}".format(""))
    #Compressor Variablen
    typ = 'Lim'
    dynamisch = 1
    t_attack = 0.0001
    t_release = 0.5
    L_thresh = -10
    ratio = 4
    makeup_gain = 6
    print(f"System B mit: typ = {typ}, dynamisch = {dynamisch}, t_attack = {t_attack}, t_release = {t_release}, L_thresh = {L_thresh}, ratio = {ratio}, makeup_gain = {makeup_gain}")
    # Klirrfaktor und THD
    output_waveformB, sr_waveformB = process_systemB(input_waveform, sr_waveform, typ, dynamisch, t_attack, t_release, L_thresh, ratio, makeup_gain, show_plot=False)
    plot_waveforms(t_waveform, input_waveform, output_waveformB, max_time=0.08)
    klirr = calc_klirr(output_waveformB, sr_waveformB, 10, height_threshold=0.01) # Der Klirrfaktor gibt einen Graph mit den erkannten Peaks zur Überprüfung dieser aus
    print(f"Klirrfaktor: {klirr:.2f}%")


    print("{0:–>17} III: Klangliche Beurteilung und Vergleich {0:–<17}".format(""))
    #Audiodatei durch System A
    input_file = input_file / np.max(np.abs(input_file)) # Normalisierung Eingangssignal
    output_fileA = process_systemA(input_file, a, b, AP, show_plot=False) #Processing und Plotten der Kennlinie
    #output_fileA = output_fileA/np.max(np.abs(output_fileA)) # Normalisierung bei klirrenden Einstellungen
    plot_waveforms(t_file, input_file, output_fileA, max_time=0.02)       #Ein- und Ausgangssignale plotten und Verstärkungsmessung

    #Vorher/Nachher ausgeben
    play_audio(input_file, sr_file)
    play_audio(output_fileA, sr_file)

    #Audiodatei durch System B
    # Processing und Plotting der Ein- und Ausgangssignale
    input_file = input_file / np.max(np.abs(input_file))
    output_fileB, sr_fileB = process_systemB(input_file, sr_file, typ, dynamisch, t_attack, t_release, L_thresh,
                                                ratio, makeup_gain)
    plot_waveforms(t_file, input_file, output_fileB,
                   max_time=len(input_file))  # Ein- und Ausgangssignale plotten und Verstärkungsmessung

    #Vorher/Nachher ausgeben
    play_audio(input_file, sr_file)
    play_audio(output_fileB, sr_fileB)

    '''
    # Speichern der verarbeiteten Datei
    save_path = "processed_audio_signal.wav"
    save_audio(save_path, output_signal, sample_rate, subtype='PCM_16')
    print(f"Das verarbeitete Signal wurde als {save_path} gespeichert.")
    '''