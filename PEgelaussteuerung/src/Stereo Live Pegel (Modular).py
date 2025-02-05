import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import platform

if platform.system() == 'Windows':
    plt.switch_backend('TkAgg')
if platform.system() == 'Darwin':
    plt.switch_backend('macosx')

def calc_energy(file, fs = 44100):
    return np.sum(file**2)/fs

# Berechnet Scheitelfaktor, RMS und Crest-Faktor
def calculate_rms(waveform):
    return np.sqrt(np.mean(np.square(waveform)))

def corr_factor(signal1, signal2, fs = 44100):
    return (np.sum(signal1 * signal2) / fs) / np.sqrt(calc_energy(signal1) * calc_energy(signal2))


# Lädt Audiodateien und teilt sie in linke und rechte Kanäle auf
def load_wavs(file_paths):
    audio_data = []
    sample_rates = []

    for file_path in file_paths:
        data, sample_rate = sf.read(file_path)
        left_channel, right_channel = split_channels(data)
        audio_data.append((left_channel, right_channel))
        sample_rates.append(sample_rate)

    ensure_same_sample_rate(sample_rates)
    audio_data = trim_audio_to_min_length(audio_data)

    return audio_data, sample_rates[0]


# Splits the audio data into left and right channels
def split_channels(data):
    if len(data.shape) == 1:
        return data, data  # Mono to both channels
    return data[:, 0], data[:, 1]


# Überprüft, ob alle Sample-Raten gleich sind
def ensure_same_sample_rate(sample_rates):
    if len(set(sample_rates)) > 1:
        raise ValueError("Alle Dateien müssen die gleiche Sample-Rate haben.")


# Kürzt die Audiodaten auf die minimale Länge
def trim_audio_to_min_length(audio_data):
    min_length = min(len(track[0]) for track in audio_data)
    return [(track[0][:min_length], track[1][:min_length]) for track in audio_data]


# Normalisiert die Audiodaten
def normalize_audio(audio_data, normalize_to=0.1):
    # Step 1: Calculate the RMS value for each track
    rms_values = [calculate_rms(track[0] + track[1]) for track in audio_data]

    # Step 2: Determine the maximum RMS value among all tracks
    max_rms = max(rms_values)

    # Step 3: Calculate the normalization factor based on the maximum RMS value
    if max_rms > 0:
        normalization_factor = 1.0 / max_rms

        # Step 4: Normalize each track using the calculated normalization factor
        normalized_audio_data = [(track[0] * normalization_factor, track[1] * normalization_factor) for track in
                                 audio_data]

        # Step 5: Combine the normalized tracks into a master output
        combined_audio_left = np.sum([track[0] for track in normalized_audio_data], axis=0)
        combined_audio_right = np.sum([track[1] for track in normalized_audio_data], axis=0)

        # Step 6: Normalize the master output to 0 dB
        master_rms = calculate_rms(combined_audio_left + combined_audio_right)
        if master_rms > 0:
            master_normalization_factor = 1.0 / master_rms
            combined_audio_left *= master_normalization_factor
            combined_audio_right *= master_normalization_factor


        return normalized_audio_data, np.column_stack((combined_audio_left, combined_audio_right))

    return audio_data, None


# Wandelt Amplitude in dBFS um
def amplitude_to_dbfs(amplitude):
    return 20 * np.log10(np.clip(amplitude, 1e-10, 1.0))


# Initialisiert die Balken für die Pegelanzeige
def initialize_bars(ax, num_tracks):
    bar_width = 0.2
    offset = 0.11
    bar_container_left = ax.bar(np.arange(num_tracks) - offset, [0] * num_tracks, width=bar_width, bottom=-60, label='Links')
    bar_container_right = ax.bar(np.arange(num_tracks) + offset, [0] * num_tracks, width=bar_width, bottom=-60, label='Rechts', color='orange')

    master_bar_left = ax.bar(num_tracks - offset, 0, width=bar_width, bottom=-60, label='Master', color='red')
    master_bar_right = ax.bar(num_tracks + offset, 0, width=bar_width, bottom=-60, color='red')

    return bar_container_left, bar_container_right, master_bar_left, master_bar_right


# Fügt X-Achsen-Beschriftungen hinzu
def set_x_labels(ax, num_tracks):
    ax.set_xticks(np.arange(num_tracks + 1))
    ax.set_xticklabels(['Track ' + str(i + 1) for i in range(num_tracks)] + ['Master'])


# Aktualisiert die Pegelanzeige
def update_meter(frame, audio_data, bar_container_left, bar_container_right, master_bar_left, master_bar_right, sample_rate):
    current_frames = [(track[0][frame:frame + sample_rate // 5], track[1][frame:frame + sample_rate // 5]) for track in audio_data]

    peak_levels_left = [amplitude_to_dbfs(np.max(calculate_rms(track[0]))) for track in current_frames]
    peak_levels_right = [amplitude_to_dbfs(np.max(calculate_rms(track[0]))) for track in current_frames]

    master_level_left = amplitude_to_dbfs(np.max(calculate_rms(np.sum([track[0] for track in current_frames], axis=0))))
    master_level_right = amplitude_to_dbfs(np.max(calculate_rms(np.sum([track[1] for track in current_frames], axis=0))))

    update_bars(bar_container_left, peak_levels_left)
    update_bars(bar_container_right, peak_levels_right)

    master_bar_left[0].set_height(master_level_left - (-60))
    master_bar_right[0].set_height(master_level_right - (-60))

    return [*bar_container_left, *bar_container_right, master_bar_left[0], master_bar_right[0]]


# Aktualisiert die Höhen der Balken
def update_bars(bar_container, peak_levels):
    for i, bar in enumerate(bar_container):
        bar.set_height(peak_levels[i] - (-60))
        bar.set_y(-60)


# Spielt das Audio in einem separaten Thread ab
def play_audio(audio_data, sample_rate):
    #combined_audio = combine_audio(audio_data)
    sd.play(audio_data, sample_rate)
    sd.wait()

def normalize_channel_rms(channel, target_rms=0.1):
    # Calculate the RMS value of the channel
    rms_value = calculate_rms(channel)

    # Determine the normalization factor
    normalization_factor = target_rms / rms_value if rms_value != 0 else 0

    # Normalize the channel
    normalized_channel = channel * normalization_factor

    return normalized_channel

# Setzt alles zusammen und führt das Programm aus
def main():

    file_paths = ["track1.wav", "track2.wav", "track3.wav", "track4.wav", "track5.wav", "track6.wav", "track7.wav", "track8.wav"]
    audio_data, sample_rate = load_wavs([os.path.join(os.path.dirname(__file__), path) for path in file_paths])

    audio_data, combined_audio = normalize_audio(audio_data, 0.1)

    print(f"Energy of track 1: {calc_energy(audio_data[0][0], sample_rate):.2f}\nEnergy of track 2: {calc_energy(audio_data[1][0], sample_rate):.2f}\nEnergy of both Tracks: {calc_energy(audio_data[0][0] + audio_data[1][0] , sample_rate):.2f}")
    print(f"Peak of Track 1: {calculate_rms(audio_data[0][0]):.2f}, Peak of Track 2: {calculate_rms(audio_data[1][0]):.2f}")
    print(f"Correlation factor between Track 1 and 2: {corr_factor(audio_data[0][0], audio_data[1][0], sample_rate):.2f}")

    audio_thread = threading.Thread(target=play_audio, args=(combined_audio, sample_rate))
    audio_thread.start()

    fig, ax = plt.subplots()
    ax.set_ylim(-60, 0)
    ax.set_title("Live Pegelanzeige")
    ax.set_ylabel("dBFS")
    ax.set_xlabel("Spuren")

    num_tracks = len(audio_data)
    bar_container_left, bar_container_right, master_bar_left, master_bar_right = initialize_bars(ax, num_tracks)
    set_x_labels(ax, num_tracks)

    ani = FuncAnimation(fig, update_meter, frames=range(0, len(audio_data[0][0]), sample_rate // 5),
                        fargs=(audio_data, bar_container_left, bar_container_right, master_bar_left, master_bar_right, sample_rate), blit=True)

    plt.legend()
    plt.show()

    audio_thread.join()


if __name__ == "__main__":
    main()
