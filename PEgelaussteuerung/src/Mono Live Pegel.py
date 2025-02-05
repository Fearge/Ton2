import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading


# Scheitelfaktor berechnen
def calculate_peak_rms(waveform):
    peak_value = np.max(np.abs(waveform))
    rms_value = np.sqrt(np.mean(np.square(waveform)))
    crest_factor = peak_value / rms_value
    return peak_value, rms_value, crest_factor


# Audiodateien laden und normalisieren
def load_wavs(file_paths):
    audio_data = []
    sample_rates = []

    for file_path in file_paths:
        data, sample_rate = sf.read(file_path)

        # Wenn Stereodatei, Kanäle auf eine Spur summieren
        if len(data.shape) > 1:
            data = np.sum(data, axis=1) / data.shape[1]

        audio_data.append(data)
        sample_rates.append(sample_rate)

    # Sicherstellen, dass alle Dateien die gleiche Sample-Rate haben
    if len(set(sample_rates)) > 1:
        raise ValueError("Alle Dateien müssen die gleiche Sample-Rate haben.")

    # Auf die gleiche Länge kürzen (Länge der kürzesten Spur)
    min_length = min(len(track) for track in audio_data)
    audio_data = [track[:min_length] for track in audio_data]

    return audio_data, sample_rates[0]


# Normalisiere die Audiodaten auf Basis der gewünschten Maximal-Peak-Amplitude (z.B. 0dBFS)
def normalize_audio(audio_data):
    # Maximalwert der Summenspur finden und als Referenz für Normalisierung nutzen
    max_sum_peak = np.max(np.abs(np.sum(audio_data, axis=0)))
    if max_sum_peak > 0:
        normalization_factor = 1 / max_sum_peak
        audio_data = [track * normalization_factor for track in audio_data]
    return audio_data


# dBFS Berechnung
def amplitude_to_dbfs(amplitude):
    return 20 * np.log10(np.clip(amplitude, 1e-10, 1.0))


# Wiedergabe und Pegelanzeige
def playback_with_meter(audio_data, sample_rate):
    # Startet Wiedergabe und erstellt Live-Pegelanzeige
    fig, ax = plt.subplots()
    ax.set_ylim(-60, 0)  # Anpassen der y-Achse von -60 bis 0 dBFS
    ax.set_title("Live Pegelanzeige")
    ax.set_ylabel("dBFS")
    ax.set_xlabel("Spuren")

    # Pegelanzeige mit Beginn bei -60 dBFS
    bar_container = ax.bar(range(len(audio_data) + 1), [0] * (len(audio_data) + 1), bottom=-60)

    # Bestimme die x-Positionen für die Teilstriche und Beschriftungen
    x_ticks = range(len(audio_data) + 1)  # Die x-Positionen für die Balken
    x_labels = ['Track ' + str(i + 1) for i in range(len(audio_data))] + ['Master']

    # Setze die Teilstriche und die Beschriftungen
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    def update_meter(frame):
        # Berechne momentane Amplituden für jede Spur und Master
        current_frames = [track[frame:frame + sample_rate // 10] for track in audio_data]
        peak_levels = [amplitude_to_dbfs(np.max(np.abs(track))) for track in current_frames]

        # Master-Level als Summe aller Spuren
        master_level = amplitude_to_dbfs(np.max(np.abs(np.sum(current_frames, axis=0))))

        # Aktualisiere Pegelanzeige
        for i, bar in enumerate(bar_container):
            if i < len(audio_data):
                # Setze die Höhe entsprechend der dBFS-Werte von -60 bis zum aktuellen Wert
                height = peak_levels[i] - (-60)  # Höhe berechnen
                bar.set_height(height)  # Setze die Höhe
                bar.set_y(-60)  # Basis bleibt bei -60 dBFS
            else:
                # Für den Master-Level
                height = master_level - (-60)  # Höhe berechnen
                bar.set_height(height)  # Setze die Höhe
                bar.set_y(-60)  # Basis bleibt bei -60 dBFS

        return bar_container

    # Animation für Live-Update
    ani = FuncAnimation(fig, update_meter, frames=range(0, len(audio_data[0]), sample_rate // 5), blit=True)
    plt.show()


# Audio-Wiedergabe in separatem Thread
def play_audio(audio_data, sample_rate):
    combined_audio = np.sum(audio_data, axis=0)
    # Normalisiere auf [-1, 1] und wandle in float32 um
    combined_audio = (combined_audio / np.max(np.abs(combined_audio))).astype(np.float32)

    sd.play(combined_audio, sample_rate)
    sd.wait()  # Warten, bis die Wiedergabe abgeschlossen ist


# Beispielaufruf
file_paths = ["track1.wav", "track2.wav", "track3.wav", "track4.wav", "track5.wav", "track6.wav", "track7.wav"]
audio_data, sample_rate = load_wavs([os.path.join(os.path.dirname(__file__), path) for path in file_paths])
audio_data = normalize_audio(audio_data)

# Starte Audio in separatem Thread, Pegelanzeige im Hauptthread
audio_thread = threading.Thread(target=play_audio, args=(audio_data, sample_rate))
audio_thread.start()

# Pegelanzeige läuft im Hauptthread
playback_with_meter(audio_data, sample_rate)

audio_thread.join()
