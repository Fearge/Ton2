import os

import numpy as np
import matplotlib.pyplot as plt
import scipy
import soundfile as sf  # Zum Dateien einlesen
from matplotlib.ticker import ScalarFormatter
from scipy.fft import fft, fftfreq


def plot_frequency_spectrum(signal, sample_rate,  n_fft=1024):
    # Time axis for the signal
    time = np.arange(len(signal)) / sample_rate

    # Calculate the FFT of the signal
    fft_magnitude = np.fft.fft(signal, n = n_fft)
    fft_magnitude = fft_magnitude[:len(fft_magnitude) // 2]  # Only take the first half of the spectrum
    freqs = np.linspace(20, sample_rate, len(fft_magnitude))

    # Convert magnitude to dB
    fft_magnitude_db = 20 * np.log10(fft_magnitude)

    # Compute the frequency bins
    #freqs = np.fft.fftfreq(n_fft, 1 / sample_rate)
    #freqs = freqs[:len(freqs) // 2]  # Only take the first half of the spectrum

    # Plot the time-domain signal
    plt.plot(time, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Time-Domain Signal')
    plt.grid()

    # Plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_magnitude_db)
    plt.xlabel('Frequency (Hz)')
    plt.xscale('log')
    #plt.xlim(20, 20000)  # Set x-axis range from 20 Hz to 20 kHz
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum')
    plt.grid()
    plt.show()

def load_wav(file_path):
    """file = os.path.join(os.path.dirname(__file__), file_path)
    data, sample_rate = sf.read(file)  # normiert Datei auf Wertebereich [-1, 1]
    #sample_rate, data = wavfile.read(file)

    mono = np.mean(data, axis=1)

    return mono, sample_rate"""

    file = os.path.join(os.path.dirname(__file__), file_path)
    data, sample_rate = sf.read(file)

    if data.ndim > 1:
        data = data[:, 0]

    return data, sample_rate


def spec(file, sample_rate):
    ft = np.fft.rfft(file * np.hanning(len(file)))
    freqs = np.linspace(0, sample_rate / 2, len(ft))
    mg_db = 20 * np.log10(abs(ft))

    # Plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, mg_db)
    plt.xlabel('Frequency (Hz)')
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.xlim(20, 22050)  # Set x-axis range from 20 Hz to 20 kHz
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum')
    plt.grid()
    plt.show()

# Example usage
if __name__ == "__main__":
    h_data, sample_rate = load_wav("test_h_von_t_W23.wav")
    duration = 1.0  # seconds
    frequency = 100  # Hz
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * frequency * t)
    spec(h_data, sample_rate)
    #plot_frequency_spectrum(h_data, sample_rate, n_fft=len(h_data))