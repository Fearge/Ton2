import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

# Wellenformen erzeugen
def generate_waveform(wave_type, frequency, duration, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    if wave_type == 'sine':
        waveform = np.sin(2 * np.pi * frequency * t)
    elif wave_type == 'square':
        waveform = signal.square(2 * np.pi * frequency * t)
    elif wave_type == 'sawtooth':
        waveform = signal.sawtooth(2 * np.pi * frequency * t)
    elif wave_type == 'triangle':
        waveform = signal.sawtooth(2 * np.pi * frequency * t, 0.5)
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


def process_signal(input_signal, a, b, work_point):
    """
    Prozessiert ein Eingangssignal mit der Kennlinie y1(x) = e^(a * x) - b.

    Parameters:
        input_signal (array-like): Eingangssignal (x-Werte).
        a (float): Parameter a der Kennlinie.
        b (float): Parameter b der Kennlinie.

    Returns:
        output_signal (array-like): Ausgangssignal (y-Werte).
    """
    # Plot der Kennlinie
    x_values_for_curve = np.linspace(min(input_signal), max(input_signal), 500)
    y_values_for_curve = np.exp(a * (x_values_for_curve - work_point)) - b
    plt.figure(figsize=(8, 6))
    plt.plot(x_values_for_curve, y_values_for_curve, color="blue", label=r'Kennlinie: $y_1(x) = e^{a \cdot (x - x_{\text{AP}})} - b$')
    plt.axhline(0, color="black", linewidth=1, linestyle="-")  # Nullachse
    plt.axvline(0, color="black", linewidth=1, linestyle="-")  # y-Achse
    plt.title('Kennlinie mit Arbeitspunkt')
    plt.xlabel('Eingangssignal')
    plt.ylabel('Ausgangssignal')
    plt.grid(True)
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

    plt.figure(figsize=(12, 6))
    plt.plot(time, input_signal, label='Eingangssignal', color='blue', alpha=0.7)
    plt.plot(time, output_signal, label='Ausgangssignal', color='red', alpha=0.7)
    plt.title('Ein- und Ausgangssignal (Waveforms)')
    plt.axhline(0, color="black", linewidth=1, linestyle="-")  # Nullachse
    plt.xlabel('Zeit [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()
    plt.show()


def play_audio(audio_data, sample_rate):
    sd.play(audio_data, sample_rate)
    sd.wait()



# Hauptteil
if __name__ == "__main__":
    a = 2
    b = +1.5
    AP = 0  # Arbeitspunkt (verschiebt Kennlinie nach links)
    print(f"a = {a}, b = {b}, AP = {AP}")

    '''
    #Testfrequenz (zum Testen mit einem freiwählbaren Sinus oder anderen Wellenformen)
    frequency = 400  # Frequenz in Hz
    duration = 2  # Dauer in Sekunden
    sample_rate = 44100  # Abtastrate
    input_signal, t = generate_waveform('sine', frequency, duration, sample_rate)
    '''

    # Sprachsignal laden
    input_signal, sample_rate = load_wav("Sprechen_1.wav")

    # Zeitvektor erstellen
    t = np.linspace(0, len(input_signal) / sample_rate, num=len(input_signal), endpoint=False)


    # Signal verarbeiten
    output_signal = process_signal(input_signal, a, b, AP)

    # Ein- und Ausgangssignale plotten
    plot_waveforms(t, input_signal, output_signal, max_time=0.02)

    #Vorher/Nachher ausgeben
    play_audio(input_signal, sample_rate)
    play_audio(output_signal, sample_rate)

    # Speichern der verarbeiteten Datei
    save_path = "processed_speech_signal.wav"
    save_audio(save_path, output_signal, sample_rate, subtype='PCM_16')
    print(f"Das verarbeitete Signal wurde als {save_path} gespeichert.")
