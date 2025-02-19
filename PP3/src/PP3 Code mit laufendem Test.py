'''
Ton2 – PP3 – Gruppe 4
Linus Ollmann
Mario Seibert
Tom Koryciak
Gabriel Christoforidis
'''

import os
import numpy as np
import scipy.signal as sg
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import platform
import time
import pandas as pd
from scipy.interpolate import make_interp_spline

# Lauffähigkeit auf allen Plattformen sicherstellen
if platform.system() == 'Windows':
    plt.switch_backend('TkAgg')
if platform.system() == 'Darwin':
    plt.switch_backend('macosx')


# Basisfunktionen für Einzahlwerte
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
        data, sample_rate = sf.read(file_path) # normiert Datei auf Wertebereich [-1, 1]
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


def normalize_audio(audio_data, correlation):
    n = len(audio_data) # Anzahl der Tracks
    print("{0:–>15} Pegelkorrektur {0:–<15}".format(""))
    print("Angenommene Korrelation:", correlation)
    print(f"Maximaler Pegel vor Normierung: = {amplitude_to_dbfs(np.max(np.abs(audio_data))):.2f} dBFS")

    if correlation == 1:
        delta_dbfs = -(20 * np.log10(n))
    elif correlation == 0:
        delta_dbfs = -(10 * np.log10(n))

    delta_ampl = 10 ** (delta_dbfs / 20)
    print(f"Pegeländerung um {delta_dbfs:.2f} dBFS")

    normalized_audio_data = [(track[0] * delta_ampl, track[1] * delta_ampl) for track in audio_data]
    print(f"Maximaler Pegel nach Normierung: {amplitude_to_dbfs(np.max(np.abs(normalized_audio_data))):.2f} dBFS")
    return normalized_audio_data


def adjust_volume_and_panning(audio_data):
    num_tracks = len(audio_data)
    print(f"\nEs gibt {num_tracks} Spuren im aktuellen Projekt.")
    print("Einstellungen für Lautstärke und Panning:")

    # Eingabe der Lautstärke für alle Tracks auf einmal
    volume_input = input(
        f"Geben Sie die Lautstärke für alle {num_tracks} Spuren ein (-inf bis 0 dBFS, getrennt durch Kommas):\n")
    volume_values = [float(v.strip()) for v in volume_input.split(",")] if volume_input else [0] * num_tracks

    if len(volume_values) != num_tracks:
        raise ValueError(
            f"Die Anzahl der Lautstärkewerte ({len(volume_values)}) stimmt nicht mit der Anzahl der Spuren ({num_tracks}) überein.")

    # Eingabe des Pannings für alle Tracks auf einmal
    panning_input = input(
        f"Geben Sie das Panning für alle {num_tracks} Spuren ein (-1 = links, 0 = Mitte, 1 = rechts, getrennt durch Kommas):\n")
    panning_values = [float(p.strip()) for p in panning_input.split(",")] if panning_input else [0] * num_tracks

    if len(panning_values) != num_tracks:
        raise ValueError(
            f"Die Anzahl der Panningwerte ({len(panning_values)}) stimmt nicht mit der Anzahl der Spuren ({num_tracks}) überein.")

    # Anwenden der Lautstärke- und Panning-Werte
    adjustments = []
    for i, (left, right) in enumerate(audio_data):
        volume_gain = dbfs_to_amplitude(volume_values[i])

        left = left * volume_gain * (1 - panning_values[i] if panning_values[i] > 0 else 1)
        right = right * volume_gain * (1 + panning_values[i] if panning_values[i] < 0 else 1)

        adjustments.append((left, right))

    return adjustments


def amplitude_to_dbfs(amplitude):
    return 20 * np.log10(np.clip(amplitude, 1e-10, 100))

def dbfs_to_amplitude(dbfs):
    return 10 ** (dbfs / 20)

'''Live Anzeige'''
# Initialisiert die Balken für die Pegelanzeige
def initialize_bars(ax, num_tracks):
    bar_width = 0.2
    offset = 0.11
    bar_container_left = ax.bar(np.arange(num_tracks) - offset, [0] * num_tracks, width=bar_width, bottom=-60, label='Links')
    bar_container_right = ax.bar(np.arange(num_tracks) + offset, [0] * num_tracks, width=bar_width, bottom=-60, label='Rechts', color='orange')

    master_bar_left = ax.bar(num_tracks - offset, 0, width=bar_width, bottom=-60, label='Master', color='red')
    master_bar_right = ax.bar(num_tracks + offset, 0, width=bar_width, bottom=-60, color='red')

    return bar_container_left, bar_container_right, master_bar_left, master_bar_right
def set_x_labels(ax, num_tracks):
    ax.set_xticks(np.arange(num_tracks + 1))
    ax.set_xticklabels(['Track ' + str(i + 1) for i in range(num_tracks)] + ['Master'])
def update_meter(frame, audio_data, bar_container_left, bar_container_right, master_bar_left, master_bar_right, sample_rate):
    current_frames = [(track[0][frame:frame + sample_rate // 5], track[1][frame:frame + sample_rate // 5]) for track in audio_data]

    peak_levels_left = [amplitude_to_dbfs(np.max(np.abs(track[0]))) for track in current_frames]
    peak_levels_right = [amplitude_to_dbfs(np.max(np.abs(track[1]))) for track in current_frames]

    master_level_left = amplitude_to_dbfs(np.max(np.abs(np.sum([track[0] for track in current_frames], axis=0))))
    master_level_right = amplitude_to_dbfs(np.max(np.abs(np.sum([track[1] for track in current_frames], axis=0))))

    '''Übersteuerung melden'''
    if master_level_left > 0 or master_level_right > 0:
        print("ÜBERSTEUERUNG!")
        print(f"L: {master_level_left:.2f} | R: {master_level_right:.2f}")

    update_bars(bar_container_left, peak_levels_left)
    update_bars(bar_container_right, peak_levels_right)

    master_bar_left[0].set_height(master_level_left - (-60))
    master_bar_right[0].set_height(master_level_right - (-60))

    return (*bar_container_left, *bar_container_right, master_bar_left[0], master_bar_right[0])
def update_bars(bar_container, peak_levels):
    for i, bar in enumerate(bar_container):
        bar.set_height(peak_levels[i] - (-60))
        bar.set_y(-60)
'''Audiowiedergabe'''
# Spielt das Audio in einem separaten Thread ab
def play_audio(audio_data, sample_rate):
    combined_audio = combine_audio(audio_data)
    sd.play(combined_audio, sample_rate)
    sd.wait()
# Kombiniert die Audiodaten in Stereo
def combine_audio(audio_data):
    combined_audio_left = np.sum([track[0] for track in audio_data], axis=0)
    combined_audio_right = np.sum([track[1] for track in audio_data], axis=0)

    combined_audio_left = normalize_channel(combined_audio_left)
    combined_audio_right = normalize_channel(combined_audio_right)

    return np.column_stack((combined_audio_left, combined_audio_right))
def normalize_channel(channel):
    return (channel / np.max(np.abs(channel))).astype(np.float32)

# Zeitlicher Korrelationsfaktor
def time_varying_correlation_mix(audio_data, fs=44100, window_size=1024, hop_size=512):
    """
    Berechnet den zeitabhängigen Korrelationsgrad zwischen der Summe der linken und rechten Kanäle der Mischung.

    Parameters:
        audio_data (list of tuples): Liste von Stereo-Audiospuren [(links, rechts), (links, rechts), ...]
        fs (int): Abtastrate (Standard: 44100 Hz)
        window_size (int): Fenstergröße für die Berechnung der Korrelation (Standard: 1024)
        hop_size (int): Schrittweite für die Fensterung (Standard: 512)

    Returns:
        numpy array: Korrelationswerte zwischen der linken und rechten Mischung (über die Zeit)
        numpy array: Zeitachse in Sekunden
    """
    num_samples = min(len(track[0]) for track in audio_data)  # Kürzt alle Spuren auf gleiche Länge
    num_frames = (num_samples - window_size) // hop_size + 1

    # Stereosumme der Mischung berechnen
    mix_left = np.sum([track[0] for track in audio_data], axis=0)
    mix_right = np.sum([track[1] for track in audio_data], axis=0)

    correlation_values = np.zeros(num_frames)
    time_axis = np.arange(num_frames) * (hop_size / fs)  # Zeitachse in Sekunden

    for i in range(num_frames):
        start = i * hop_size
        end = start + window_size

        left_window = mix_left[start:end]
        right_window = mix_right[start:end]

        # Verwenden der vorhandenen Korrelationsfunktion
        correlation_values[i] = corr_factor(left_window, right_window, fs)

    return correlation_values, time_axis

#Abb 2
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
def binaural_signal_masking():
    # Experimentparameter
    noise_trials = 5  # Anzahl der Wiederholungen für das Rausch-Experiment

    # Lade die vorhandene WAV-Datei
    audio_filename = "Audiobeispiel - Sprache.wav"
    audio, fs = sf.read(audio_filename)

    noise1 = np.random.randn(len(audio))
    noise2 = np.random.randn(len(audio))

    # Falls die Datei mono ist, duplizieren wir sie für Stereo
    if len(audio.shape) == 1:
        audio = np.column_stack((audio, audio))

    def normalization(audio):
        return audio / max(1.0, np.max(np.abs(audio)))  # Normalisierung mit Clipping-Schutz

    def play_pink_noise_experiment(condition, volume_factor, audio = audio):

        if condition == "korreliert":
            left_channel = noise1
            right_channel = noise1
        else:  # "dekorreliert"
            left_channel = noise1
            right_channel = noise2

        stereo_noise = np.column_stack((left_channel, right_channel))
        stereo_noise = normalization(stereo_noise)

        # Kürze oder verlängere das Ton-Signal auf die Länge des pinken Rauschens
        stereo_audio = np.tile(audio, (int(np.ceil(len(stereo_noise) / len(audio))), 1))[:len(stereo_noise)]
        stereo_audio *= volume_factor  # Anwenden des Lautstärkefaktors

        combined_signal = normalization(stereo_noise + stereo_audio)

        sd.play(combined_signal, samplerate=fs)
        sd.wait()

    def run_noise_experiment():
        conditions = ["dekorreliert", "korreliert"]
        np.random.shuffle(conditions)

        responses_noise = []

        for condition in conditions:
            volume_factor = 1.0
            while True:
                play_pink_noise_experiment(condition, volume_factor=volume_factor)
                response = input("War das Sprachsignal noch hörbar? (Ja/Nein): ")
                if response.lower() == "nein":
                    break
                volume_factor *= 0.8  # Verringere die Lautstärke um 20%

            # Berechne und drucke den Pegel des Audiosignals
            audio_level_db = 20 * np.log10(volume_factor)
            print(f"Pegel des Audiosignals bei dem es gerade noch hörbar war: {audio_level_db:.2f} dB")

            # Berechne und drucke den interauralen Kohärenzgrad des verwendeten Rauschens
            left_channel = noise1 if condition == "korreliert" else noise1
            right_channel = noise1 if condition == "korreliert" else noise2
            coherence = interaurale_kohaerenz(left_channel, right_channel)
            print(f"Interauraler Kohärenzgrad des verwendeten Rauschens: {coherence:.2f}")

    run_noise_experiment()

#Abb 3
def generiere_knackse(fs = 44100):
    # Anzahl der Reihen und Spalten (zeitliche Anordnung)
    rows = 5  # Reihen (Häufigkeit)
    cols = 4  # Spalten (Wiederholungen)

    # Impulsparameter
    impulsdauer = 0.001  # Sekunden
    pausenzeit = 0.1  # Zeit zwischen Impulsen

    # Erzeuge einen einzelnen Knack-Impuls
    impuls_samples = int(impulsdauer * fs)
    impuls = np.concatenate([np.ones(impuls_samples), np.zeros(int(pausenzeit * fs))])

    # Rechteckige Sequenz generieren
    knacks = np.tile(impuls, cols)  # In Spalten wiederholen

    # Mehrere Reihen erzeugen (mit Pause zwischen den Gruppen)
    pause_zwischen_reihen = np.zeros(int(0.5 * fs))  # 0.5 Sekunden Pause zwischen Reihen
    knackse = np.concatenate([knacks] * rows + [pause_zwischen_reihen])
    return np.column_stack((knackse, knackse))
def normalization(audio):
    return audio / max(1.0, np.max(np.abs(audio)))  # Normalisierung mit Clipping-Schutz
def stereo_imaging():
    # Experimentparameter
    num_trials = 15  # Anzahl der Wiederholungen
    level_range = (-18, 18)  # Pegeldifferenz in dB

    # Lade die vorhandene WAV-Datei
    audio_filename = "Audiobeispiel - Sprache.wav"
    tone, fs = sf.read(audio_filename)
    knacksignal = generiere_knackse(fs)

    # Falls die Datei mono ist, duplizieren wir sie für Stereo
    if len(tone.shape) == 1:
        tone = np.column_stack((tone, tone))

    def play_with_level_difference(signal, level_diff, duration=5):
        factor = 10 ** (level_diff / 20)  # Umrechnung von dB in linearen Faktor

        if level_diff > 0:
            left_channel = signal[:, 0] * factor
            right_channel = signal[:, 1]
        else:
            left_channel = signal[:, 0]
            right_channel = signal[:, 1] * (1 / factor)

        stereo_signal = np.column_stack((left_channel, right_channel))
        stereo_signal = normalization(stereo_signal[:int(duration * fs)])  # Begrenze und normalisiere

        sd.play(stereo_signal, samplerate=fs)
        sd.wait()

    # Einführungssignale abspielen
    print(
        "Wichtige Infos zur Korrekten Experimentdurchführung: \n- Verwendung von Stereolautsprechen statt Kopfhörern \n- Positionierung des Kopfes, dass sich zwischen den Lautsprechern mindestens ein Winkel von 60° ergibt")
    print(
        "Sie hören jetzt zuerst die maximale Rechtslokalisation (-18 dB), dann die maximale Linkslokalisation (+18 dB)")

    # Beispielsignale abspielen
    play_with_level_difference(tone, 18, duration=5)  # Maximale Linkslokalisation
    play_with_level_difference(tone, -18, duration=5)  # Maximale Rechtslokalisation

    sprache_values = set()
    sprache_level = []

    while len(sprache_values) < num_trials:
        level_diff = np.random.randint(*level_range)  # Ganzzahlige Werte
        if abs(level_diff) not in sprache_values:
            sprache_values.add(abs(level_diff))  # Betrag nehmen

            play_with_level_difference(tone, level_diff, duration=5)  # Gesamte Datei abspielen

            while True:
                response = input("Wie haben Sie das Signal wahrgenommen? (-1 = rechts, 0 = Mitte, 1 = links): ")
                response = response.replace(",", ".")  # Ersetze Komma durch Punkt
                try:
                    response_value = float(response)
                    if -1 <= response_value <= 1:
                        sprache_level.append(abs(response_value) * 100)  # Betrag nehmen und in Prozent umwandeln
                        break
                    else:
                        print("Bitte geben Sie einen Wert zwischen -1 und 1 ein.")
                except ValueError:
                    print("Ungültige Eingabe. Bitte eine Zahl zwischen -1 und 1 eingeben.")

    # Test mit Knacksignal
    print("Jetzt folgt der Test mit dem Knacksignal.")
    play_with_level_difference(knacksignal, 18, duration=5)  # Maximale Rechtslokalisation
    play_with_level_difference(knacksignal, -18, duration=5)  # Maximale Linkslokalisation

    knack_values = set()
    knack_levels = []

    while len(knack_values) < num_trials:
        level_diff = np.random.randint(*level_range)  # Ganzzahlige Werte
        if abs(level_diff) not in knack_values:
            knack_values.add(abs(level_diff))  # Betrag nehmen

            play_with_level_difference(knacksignal, level_diff, duration=5)  # Gesamte Datei abspielen

            while True:
                response = input("Wie haben Sie das Knacksignal wahrgenommen? (-1 = rechts, 0 = Mitte, 1 = links): ")
                response = response.replace(",", ".")  # Ersetze Komma durch Punkt
                try:
                    response_value = float(response)
                    if -1 <= response_value <= 1:
                        knack_levels.append(abs(response_value) * 100)  # Betrag nehmen und in Prozent umwandeln
                        break
                    else:
                        print("Bitte geben Sie einen Wert zwischen -1 und 1 ein.")
                except ValueError:
                    print("Ungültige Eingabe. Bitte eine Zahl zwischen -1 und 1 eingeben.")

    # Sortiere die Paare nach den jeweiligen Differenzen
    sprache_sorted = sorted(zip(sprache_values, sprache_level),
                          key=lambda x: x[0])  # Sortierung nach Pegeldifferenz

    # Entpacken der sortierten Paare
    sprache_x, sprache_y = zip(*sprache_sorted)  # X: Pegeldifferenz, Y: Wahrgenommene Richtung

    # Plot der Ergebnisse
    plt.figure(figsize=(10, 5))
    # Pegeldifferenz-Plot
    plt.subplot(1, 2, 1)
    plt.scatter(sprache_x, sprache_y, label="Messwerte", color='r')  # Scatterplot der Punkte

    # Trendlinie berechnen und plotten
    z = np.polyfit(sprache_x, sprache_y, 3)
    p = np.poly1d(z)
    plt.plot(sprache_x, p(sprache_x), "b--", label="Trendlinie")

    plt.xlabel("Pegeldifferenz (dB)")
    plt.ylabel("Wahrgenommene Richtung (%)")
    plt.title("Pegeldifferenz-Stereofonie")
    plt.legend()

    # Sortiere die Paare nach den jeweiligen Differenzen
    knack_sorted = sorted(zip(knack_values, knack_levels),
                          key=lambda x: x[0])  # Sortierung nach Pegeldifferenz
    # Entpacken der sortierten Paare
    knack_x, knack_y = zip(*knack_sorted)  # X: Pegeldifferenz, Y: Wahrgenommene Richtung

    # Pegeldifferenz-Plot
    plt.subplot(1, 2, 2)
    plt.scatter(knack_x, knack_y, label="Messwerte", color='r')  # Scatterplot der Punkte

    # Trendlinie berechnen und plotten
    z = np.polyfit(knack_x, knack_y, 3)
    p = np.poly1d(z)
    plt.plot(knack_x, p(knack_x), "b--", label="Trendlinie")

    plt.xlabel("Pegeldifferenz (dB)")
    plt.ylabel("Wahrgenommene Richtung (%)")
    plt.title("Pegeldifferenz-Stereofonie")
    plt.legend()

    # Layout anpassen und anzeigen
    plt.tight_layout()
    plt.show()

#Abb 1
#Mischung des Signals, abhängig von der Kohärenz
def mix_signals(signal1, signal2, coherence):
    alpha = np.sqrt((1 - coherence) / (1 + coherence))

    mixed_left = signal1 + alpha * signal2
    mixed_right = signal1 - alpha * signal2

    return mixed_left, mixed_right
def play_noise_with_coherence(duration, coherence, fs=44100, ampl=1.0):
    noise1 = np.random.randn(int(duration * fs))
    noise2 = np.random.randn(int(duration * fs))

    left, right = mix_signals(noise1, noise2, coherence)

    stereo_signal = np.column_stack((left, right))
    stereo_signal = normalization(stereo_signal)
    stereo_signal *= ampl
    sd.play(stereo_signal, samplerate=fs)
    sd.wait()
    return left, right
def play_randomized_coherence_noises():
    coherences = [1, 1, 0.85, 0.85, 0.4, 0.4,  0, 0,]
    measuered_coherences = []
    np.random.shuffle(coherences)
    print(f"Es werden randomisiert {len(coherences)} Rauschsignale mit unterschiedlichen Kohärenzgraden abgespielt.\n "
          "Bitte notieren Sie, wo das jeweilige Höreignis wahrgenommen wird")


    for i, coherence in enumerate(coherences):
        print(f"Darbietung: {i+1}")
        left, right = play_noise_with_coherence(5, coherence)
        measuered_coherences.append(f"{interaurale_kohaerenz(left, right): .2f}")
        time.sleep(5)

    print("{:<15} {:<15}".format("Darbietung", "Kohärenzgrad"))
    print("-" * 30)
    for i, coherence in enumerate(measuered_coherences):
        print("{:<15} {:<15}".format(i + 1, coherence))


if __name__ == "__main__":
    """knackse = generiere_knackse(44100)
    sd.play(knackse, 44100)
    sd.wait()"""
    '''
    file_paths = ["track1.wav", "track2.wav", "track3.wav", "track4.wav", "track5.wav", "track6.wav", "track8.wav"]
    audio_data, sample_rate = load_wavs([os.path.join(os.path.dirname(__file__), path) for path in file_paths])
    audio_data = normalize_audio(audio_data, 0)

    # Lautstärke und Panning anpassen
    audio_data = adjust_volume_and_panning(audio_data)

    audio_thread = threading.Thread(target=play_audio, args=(audio_data, sample_rate))
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

    J# Berechnung der zeitabhängigen Korrelation
    correlation_values, time_axis = time_varying_correlation_mix(audio_data, fs=44100)

    # Plot der zeitabhängigen Korrelation
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, correlation_values, label="Korrelationsgrad (L ↔ R)")
    plt.xlabel("Zeit (s)")
    plt.ylabel("Korrelationsgrad")
    plt.title("Zeitabhängige Korrelation zwischen linkem und rechtem Mischkanal")
    plt.legend()
    plt.show()
    '''
    # überprüfung zu Abb. 1
    #play_randomized_coherence_noises()
    # Überprüfung zu Abb.2
    #binaural_signal_masking()
    # Überprüfung zu Abb.3
    stereo_imaging()