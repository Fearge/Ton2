# ---- Ton2 WiSe24/25 PA1 Gruppe 3 ----

# Audiogram.py


import sys
import threading
import time
from threading import Thread

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import keyboard

fs = 44100
testTone = 1000
duration = 0.5
volume = 0.00001
volume_increment = volume
delayTime = 1

level_of_hearing = []
frequency_to_test = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=2, rate=fs, output=True)

def input_thread(a_list):
    input()
    a_list.append(True)

def generate_sine(frequency, dur):
    t = np.linspace(0, dur, int(fs * dur), True)
    # generate sine wave
    samples = (np.sin(2*np.pi * frequency * t )).astype(np.float32)
    return samples

def generate_noise(vol):
    # generate noise
    samples = np.random.uniform(-vol, vol, int(fs * 0.5)).astype(np.float32)
    return samples

def play_tone(freq, dur, vol):
    #play sine
    sd.play(vol * generate_sine(freq, dur), fs)
    time.sleep(delayTime)
    sd.wait()

def play_noise(vol):
    while True:
        stream.write(generate_noise(vol).tobytes())

def check_keyboard():
    #check if user presses a key
    return keyboard.is_pressed("h")


def hear_test(freq, vol = volume):
    a_list = []

    keyboard_thread = threading.Thread(target=input_thread, args=(a_list,))
    keyboard_thread.start()
    #increment volume of each test frequency until user presses a key
    print("Drücke Enter beim Hören des Tons")
    while volume < 1:
        play_tone(freq, duration, vol)
        if a_list:
            break
        vol += volume_increment
    return vol

def plot_results(levels):
    global frequency_to_test
    levels = np.array(levels)
    frequencies = np.array(frequency_to_test)
    levels_ohne_Verdeckung = np.array(levels[0:8])
    levels_mit_Verdeckung = levels[8:16]


    print(frequency_to_test)
    # plot data with connection between dots
    plt.plot(frequencies, levels_ohne_Verdeckung, '-o', color='blue',label='Ohne Verdeckung')
    plt.plot(frequencies, levels_mit_Verdeckung, '-o', color='orange',label='Mit Verdeckung')

    # set x axis logarithmic
    plt.xscale('log')

    # set x axis ticks
    plt.xticks(frequencies, labels=[str(f) for f in frequencies])

    # title + labels
    plt.xlabel('Frequenz [Hz]')
    plt.ylabel('Pegel in Relation zu Hörschwelle bei 1kHz [dB]')
    plt.title('Audiogramm')
    plt.legend()

    plt.grid(True,which='both', linestyle='--', linewidth=0.5)

    plt.show()

def main(isNoise = False):
    #await play_noise(0.1)
    print("Start Audiogramm Test")
    for f in frequency_to_test:
        print("Testing Frequency: ", f)
        level_of_hearing.append(10 * np.log10(hear_test(f) / ref))

    if not isNoise:
        noise_thread = Thread(target=play_noise, args=(ref, ))
        noise_thread.start()
        print("Starte Test mit Verdeckung")
        time.sleep(2)
        # recall main mit Verdeckung
        main(True)
    else:
        plot_results(level_of_hearing)
        stream.stop_stream()
        stream.close()
        p.terminate()
        sys.exit()
        #noise_thread.join()


if __name__ == '__main__':
    print(
        "Set up des Referenz Pegels\n Wenn bereits der erste Ton gehört wird, "
        "bitte die Lautstärke des Abspielgeräts reduzieren")
    ref = hear_test(1000)
    print("Referenz Pegel: ", ref)
    main()




