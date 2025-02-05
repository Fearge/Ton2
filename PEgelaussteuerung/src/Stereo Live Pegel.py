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


# Audiodateien laden und in linke und rechte Kanäle aufteilen
def load_wavs(file_paths):
    audio_data = []
    sample_rates = []

    for file_path in file_paths:
        data, sample_rate = sf.read(file_path)

        if len(data.shape) == 1:
            left_channel = data
            right_channel = data  # Mono in beide Kanäle kopieren
        else:
            left_channel = data[:, 0]
            right_channel = data[:, 1]

        audio_data.append((left_channel, right_channel))  # Speichere beide Kanäle
        sample_rates.append(sample_rate)

    if len(set(sample_rates)) > 1:
        raise ValueError("Alle Dateien müssen die gleiche Sample-Rate haben.")

    # Kürze auf die gleiche Länge
    min_length = min(len(track[0]) for track in audio_data)
    audio_data = [(track[0][:min_length], track[1][:min_length]) for track in audio_data]

    return audio_data, sample_rates[0]


# Normalisiere die Audiodaten auf Basis der gewünschten Maximal-Peak-Amplitude (z.B. 0dBFS)
def normalize_audio(audio_data):
    max_sum_peak = np.max(np.abs(np.sum([track[0] + track[1] for track in audio_data], axis=0)))
    if max_sum_peak > 0:
        normalization_factor = 1 / max_sum_peak
        audio_data = [(track[0] * normalization_factor, track[1] * normalization_factor) for track in audio_data]
    return audio_data


# dBFS Berechnung
def amplitude_to_dbfs(amplitude):
    return 20 * np.log10(np.clip(amplitude, 1e-10, 1.0))


# Wiedergabe und Pegelanzeige
def playback_with_meter(audio_data, sample_rate):
    fig, ax = plt.subplots()
    ax.set_ylim(-60, 0)  # Anpassen der y-Achse von -60 bis 0 dBFS
    ax.set_title("Live Pegelanzeige")
    ax.set_ylabel("dBFS")
    ax.set_xlabel("Spuren")

    # Anzahl der Spuren
    num_tracks = len(audio_data)
    bar_width = 0.2  # Breite der Balken
    offset = 0.11  # Offset für die Balken

    # Initialisiere Balken für Links, Rechts
    bar_container_left = ax.bar(np.arange(num_tracks) - offset, [0] * num_tracks, width=bar_width, bottom=-60,
                                label='Links')  # Links
    bar_container_right = ax.bar(np.arange(num_tracks) + offset, [0] * num_tracks, width=bar_width, bottom=-60,
                                 label='Rechts', color='orange')  # Rechts

    # Master-Pegel-Balken (teilen sich einen Teilstrich)
    master_bar_left = ax.bar(num_tracks - offset, 0, width=bar_width, bottom=-60, label='Master', color='red')  # Master Links
    master_bar_right = ax.bar(num_tracks + offset, 0, width=bar_width, bottom=-60, color='red')  # Master Rechts, leicht verschoben

    # X-Achsen-Beschriftungen hinzufügen
    ax.set_xticks(np.arange(num_tracks + 1))  # Platz für beide Master auf der gleichen Stelle
    ax.set_xticklabels(['Track ' + str(i + 1) for i in range(num_tracks)] + ['Master'])

    def update_meter(frame):
        # Berechne die momentanen Frames für jeden Kanal (links und rechts)
        current_frames = [(track[0][frame:frame + sample_rate // 5], track[1][frame:frame + sample_rate // 5]) for
                          track in audio_data]

        # Berechnung der Pegel für jeden Kanal
        peak_levels_left = [amplitude_to_dbfs(np.max(np.abs(track[0]))) for track in current_frames]
        peak_levels_right = [amplitude_to_dbfs(np.max(np.abs(track[1]))) for track in current_frames]

        # Berechnung der Master-Pegel (Summe der Pegel)
        master_level_left = amplitude_to_dbfs(np.max(np.abs(np.sum([track[0] for track in current_frames], axis=0))))
        master_level_right = amplitude_to_dbfs(np.max(np.abs(np.sum([track[1] for track in current_frames], axis=0))))

        # Aktualisierung der Balken
        for i, (bar_left, bar_right) in enumerate(zip(bar_container_left, bar_container_right)):
            height_left = peak_levels_left[i] - (-60)  # Höhe für den linken Kanal
            height_right = peak_levels_right[i] - (-60)  # Höhe für den rechten Kanal
            bar_left.set_height(height_left)
            bar_right.set_height(height_right)
            bar_left.set_y(-60)  # Basis bleibt bei -60 dBFS
            bar_right.set_y(-60)  # Basis bleibt bei -60 dBFS

        # Update Master Bars
        master_bar_left[0].set_height(master_level_left - (-60))
        master_bar_left[0].set_y(-60)  # Basis bleibt bei -60 dBFS

        master_bar_right[0].set_height(master_level_right - (-60))
        master_bar_right[0].set_y(-60)  # Basis bleibt bei -60 dBFS

        # Rückgabe aller Bar-Container
        return (*bar_container_left, *bar_container_right, master_bar_left[0], master_bar_right[0])

    # Animation für Live-Update
    ani = FuncAnimation(fig, update_meter, frames=range(0, len(audio_data[0][0]), sample_rate // 5), blit=True)
    plt.legend()
    plt.show()


# Audio-Wiedergabe in separatem Thread
def play_audio(audio_data, sample_rate):
    combined_audio_left = np.sum([track[0] for track in audio_data], axis=0)
    combined_audio_right = np.sum([track[1] for track in audio_data], axis=0)

    # Normalisiere auf [-1, 1] und wandle in float32 um
    combined_audio_left = (combined_audio_left / np.max(np.abs(combined_audio_left))).astype(np.float32)
    combined_audio_right = (combined_audio_right / np.max(np.abs(combined_audio_right))).astype(np.float32)

    # Kombiniere die beiden Kanäle
    combined_audio = np.column_stack((combined_audio_left, combined_audio_right))

    sd.play(combined_audio, sample_rate)
    sd.wait()  # Warten, bis die Wiedergabe abgeschlossen ist


# Beispielaufruf
file_paths = ["track1.wav", "track2.wav", "track3.wav", "track4.wav", "track5.wav", "track6.wav"]
audio_data, sample_rate = load_wavs([os.path.join(os.path.dirname(__file__), path) for path in file_paths])
audio_data = normalize_audio(audio_data)

# Starte Audio in separatem Thread, Pegelanzeige im Hauptthread
audio_thread = threading.Thread(target=play_audio, args=(audio_data, sample_rate))
audio_thread.start()

# Pegelanzeige läuft im Hauptthread
playback_with_meter(audio_data, sample_rate)

audio_thread.join()
