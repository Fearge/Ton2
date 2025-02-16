'''
Ton2 – PP2 – Gruppe 4
Linus Ollmann
Mario Seibert
Tom Koryciak
Gabriel Christoforidis
'''

import os
import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sounddevice as sd
import soundfile as sf

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


def approximate_linear(a, b = 0, N = 16):
    """
    Approximiert eine lineare Kennlinie y = a * x + b mit einer e-Funktion der Form y = c*e^(d * x)
    anhand der Methode der kleinsten Quadrate
    Parameters
    ----------
    a = Steigung der linearen Kennlinie
    b = y-Achsenabschnitt der linearen Kennlinie
    N = Anzahl der Auflösungsbits
    dyn_range = nuerische Grenzen des Dynamikbereichs

    Returns
    -------
    c, d, arithmetischer Fehler der Approximation
    """

    # lineare Kennlinie
    x = np.linspace(1, 3, 2**N)
    y = a * x + b

    # e-Funktion
    [d,c] = np.polyfit(x, np.log(y), 1)

    # arithmetischer Fehler
    error = np.sum((y - np.exp(c) * np.exp(d * x))) / len(y)

    # beide Funktionen um den Nullpunkt plotten
    plt.figure(figsize=(8, 6))
    plt.plot(x-2, y-2, label='Lineare Kennlinie')
    plt.plot(x-2, np.exp(c) * np.exp(d * x) - 2, label='Exponentielle Kennlinie')
    plt.title('Ideale und approximierte Kennlinie der Form $y = a \cdot e^{b \cdot x}$')
    plt.xlabel('Eingangssignal')
    plt.ylabel(f'Ausgangssignal')
    plt.legend()
    plt.show()
    return c, d, error






def process_signal(input_signal, a, b, work_point):
    """
    Prozessiert ein Eingangssignal mit der Kennlinie y1(x) = e^(a * (x - work_point)) - b und plottet die Kennlinie.

    Parameters:
        input_signal (array-like): Eingangssignal (x-Werte).
        a (float): Parameter a der Kennlinie.
        b (float): Parameter b der Kennlinie.
        work_point (float): Arbeitspunkt auf der x-Achse.

    Returns:
        output_signal (array-like): Ausgangssignal (y-Werte).
    """
    # Plot der Kennlinie
    x_values_for_curve = np.linspace(-1.2, 1.2, 500)
    y_values_for_curve = np.exp(a * (x_values_for_curve - work_point)) - b

    plt.figure(figsize=(8, 6))
    plt.plot(x_values_for_curve, y_values_for_curve, color="blue",
             label=r'Kennlinie: $y_1(x) = e^{a \cdot (x - x_{\text{AP}})} - b$')

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

    plt.title('System A (Kennlinie mit Arbeitspunkt)')
    plt.xlabel('Eingangssignal')
    plt.ylabel('Ausgangssignal')
    plt.legend()
    plt.show()

    output_signal = np.exp(a * (input_signal - work_point)) - b
    return output_signal

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
    plt.plot(time, output_signal_centered, label='Ausgangssignal (DC-offset korrigiert)', color='red', alpha=0.7)

    # Verstärkungswerte als Text ins Diagramm schreiben
    text_x = time[int(len(time) * 0.85)]  # Position für das Label (85% der Zeitachse)
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

    freqs = np.linspace(0, sample_rate, len(fft_magnitude))

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_magnitude, label='Spectrum')
    plt.plot(freqs[peaks], fft_magnitude[peaks], 'ro', label='Peaks')  # Mark the peaks
    for peak in peaks:
        plt.annotate(f'{fft_magnitude[peak]:.2f}', (freqs[peak], fft_magnitude[peak]),
                     textcoords="offset points", xytext=(0, 10), ha='center')  # Annotate the peaks
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
    # returns the magnitude of the peaks of the spectrum + the spectrum
    fft_magnitude = np.abs(np.fft.fft(signal))
    fft_magnitude = fft_magnitude[:len(fft_magnitude)//2] # Only take the first half of the spectrum
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

    #print("Die Verstärkung beträgt:", f"{gain:.1f}", "\nDie Verstärkung in dB beträgt:", f"{gaindb:.1f}", "dB.")

    return gain, gaindb


#Kompressor




def play_audio(audio_data, sample_rate):
    sd.play(audio_data, sample_rate)
    sd.wait()



# Hauptteil
if __name__ == "__main__":
    print("{0:–>23} Laden der Eingangssignale {0:–<23}".format(""))
    print("<Siehe Code>")
    # Testsignal für Klirrfaktor erzeugen
    frequency = 1000  # Frequenz in Hz
    duration = 0.1  # Dauer in Sekunden
    sample_rate = 44100  # Abtastrate
    input_waveform, t = generate_waveform('sine', frequency, duration, sample_rate)

    # Audiodatei laden
    #input_signal, sample_rate = load_wav("Sprechen_1.wav")
    #t = np.linspace(0, len(input_signal) / sample_rate, num=len(input_signal), endpoint=False)



    #Kennlinienparameter
    print("{0:–>23} Einstellen der Kennlinie {0:–<23}".format(""))
    a = 2
    b = 0
    AP = 0.4  # Arbeitspunkt (verschiebt praktisch Kennlinie nach links)
    print(f"a = {a}, b = {b}, AP = {AP}")
    approximate_linear(1.1, N = 8)

    print("{0:–>23} Processing und Plotten {0:–<23}".format(""))
    print("<Siehe Plots>")
    #Verarbeitung mit der Kennlinie und Plotten der Kennlinie
    output_signal = process_signal(input_signal, a, b, AP)

    #Ein- und Ausgangssignale plotten und Verstärkungsmessung
    plot_waveforms(t, input_signal, output_signal, max_time=0.02)


    #Klirrfaktor und THD
    print("{0:–>21} Klirrfaktor und THD berechnen {0:–<21}".format(""))
    output_waveform = process_signal(input_waveform, a, b, AP)
    thd = calc_thd(output_waveform, sample_rate, 100)
    klirr = calc_klirr(output_waveform, sample_rate, 100)
    print(f"THD: {thd:.2f}%")
    print(f"Klirrfaktor: {klirr:.2f}%")




    print("{0:–>17} Vorher/Nachher abspielen und Speichern {0:–<17}".format(""))
    #Vorher/Nachher ausgeben
    play_audio(input_signal, sample_rate)
    play_audio(output_signal, sample_rate)

    # Speichern der verarbeiteten Datei
    save_path = "processed_audio_signal.wav"
    save_audio(save_path, output_signal, sample_rate, subtype='PCM_16')
    print(f"Das verarbeitete Signal wurde als {save_path} gespeichert.")
    ''''''