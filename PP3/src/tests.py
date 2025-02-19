import numpy as np
import sounddevice as sd
import scipy.signal as sg
from Tools.scripts.generate_re_casefix import alpha


def interaurale_kohaerenz(left_channel, right_channel, fs = 44100):
    """
    Berechnet die interaurale Kohärenz als Maximum der normierten Kreuzkorrelationsfunktion.

    Parameter:
    left_channel : np.array - Audiosignal für linkes Ohr
    right_channel : np.array - Audiosignal für rechtes Ohr

    Rückgabe:
    k : float - Interaurale Kohärenz (Wert zwischen 0 und 1)
    """

    max_lag_samples = int(0.001 * fs)  # Maximale Zeitverzögerung in Samples
    # Berechnung der Kreuzkorrelation
    corr = sg.correlate(left_channel, right_channel, mode='full')

    # Normierung der Kreuzkorrelation
    norm_factor = np.sqrt(np.sum(left_channel**2)* np.sum(right_channel**2))
    corr /= norm_factor  # Normierung zwischen -1 und 1

    # Begrenze auf den relevanten Bereich der Verzögerungen (± max_lag_samples)
    center = len(corr) // 2
    corr_limited = corr[center - max_lag_samples : center + max_lag_samples + 1]

    # Maximum der normierten Kreuzkorrelation bestimmen
    k = np.max(np.abs(corr_limited))

    return k

def generate_noise(duration, fs=44100):
    return np.random.randn(int(duration * fs))


def mix_signals(signal1, signal2, coherence):
    alpha = np.sqrt((1 - coherence) / (1 + coherence))

    mixed_left = signal1 + alpha * signal2
    mixed_right = signal1 - alpha * signal2

    return mixed_left, mixed_right


def play_noise_with_coherence(duration, coherence, fs=44100):
    noise1 = generate_noise(duration, fs)
    noise2 = generate_noise(duration, fs)

    left, right = mix_signals(noise1, noise2, coherence)
    print("Interaurale Kohärenz: ", interaurale_kohaerenz(left, right, fs))

    stereo_signal = np.column_stack((left, right))
    sd.play(stereo_signal, samplerate=fs)
    sd.wait()


# Beispiel: 5 Sekunden Breitbandrauschen mit einem Kohärenzgrad von 0.5 abspielen
play_noise_with_coherence(5, 0.4)