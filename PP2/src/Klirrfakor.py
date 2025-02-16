import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

# Constants
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DISTANCE = 100
DEFAULT_HEIGHT_THRESHOLD = 0.01
DEFAULT_THRESHOLD = 0.01

def generate_waveform(wave_type, frequency, duration, sample_rate=44100):
    """
    Erzeugt eine Wellenform eines bestimmten Typs, einer bestimmten Frequenz und Dauer.

    Parameter:
        wave_type (str): Typ der Wellenform ('sine', 'square', 'sawtooth', 'triangle').
        frequency (float): Frequenz der Wellenform in Hz.
        duration (float): Dauer der Wellenform in Sekunden.
        sample_rate (int): Abtastrate in Hz.

    Rückgabe:
        np.ndarray: Erzeugte Wellenform.
    """

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

    return waveform

def plot_signal(signal, frequency, sample_rate, cycles = 2):
    """
    Plottet ein Signal für eine bestimmte Anzahl von Zyklen.

    Parameter:
        signal (np.ndarray): Zu plottendes Signal.
        frequency (float): Frequenz des Signals in Hz.
        sample_rate (int): Abtastrate in Hz.
        cycles (int): Anzahl der zu plottenden Zyklen.
    """

    period = cycles / frequency
    num_samples_per_cycle = int(period * sample_rate) + 1 # plus one to get it displayed to 0 again
    one_cycle = signal[:num_samples_per_cycle]

    plt.figure(figsize=(10, 6))
    plt.plot(np.linspace(0, period, num_samples_per_cycle), one_cycle)
    plt.xlabel('Zeit (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

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

# Beispielverwendung
if __name__ == "__main__":
    sr = 44100
    freq = 1000
    signal = generate_waveform('square', freq, 0.1, sr)
    thd = calc_thd(signal, sr, 100)
    klirr = calc_klirr(signal, sr, 100)
    print(f"THD: {thd:.2f}%")
    print(f"Klirrfaktor: {klirr:.2f}%")
