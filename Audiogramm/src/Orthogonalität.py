# ---- Ton2 WiSe24/25 PA2 Gruppe 3 ----

# Orthogonalitätstest für zwei Sinussignale


import numpy as np

fs = 44100  # Abtastrate

def orthogonal_test(f1, f2, fs, duration):
    t = np.linspace(0, duration, int(fs * duration))
    signal1 = np.sin(2 * np.pi * f1 * t)
    signal2 = np.sin(2 * np.pi * f2 * t)
    orthogonality = np.sum(signal1 * signal2) /fs
    return orthogonality

# Eingabe der Frequenzen
try:
    f1 = float(input("Geben Sie die Frequenz f1 in Hz ein: "))
    f2 = float(input("Geben Sie die Frequenz f2 in Hz ein: "))
except ValueError:
    print("Bitte geben Sie gültige Zahlen für die Frequenzen ein.")
    exit()

# Test
result = orthogonal_test(f1, f2, fs, 1)
print(f"Orthogonalitätstest für f1={f1} Hz und f2={f2} Hz: rho = {result:.6f}")

#Ergebnisse
if np.isclose(result, 0, atol=1e-3):  # Toleranz für numerische Fehler
    print(f"Die Frequenzen {f1} Hz und {f2} Hz sind orthogonal.")
else:
    print(f"Die Frequenzen {f1} Hz und {f2} Hz sind nicht orthogonal.")

