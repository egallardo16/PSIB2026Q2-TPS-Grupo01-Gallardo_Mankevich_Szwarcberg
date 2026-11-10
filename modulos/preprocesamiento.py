from scipy.signal import butter, sosfiltfilt, welch
from scipy.signal.windows import tukey
import pandas as pd
import numpy as np
from pathlib import Path

def cargar_ECG_offline(codigo, fs):

  path = Path(f"registros/{codigo}.txt")
  df = pd.read_csv(path, sep=r"\s+", header=None, names=["voltaje", "tiempo", "raw"])

  df = df.drop(columns=["raw"]) #saco columna de datos crudos
  df = df.drop(columns=["tiempo"]) #saco el tiempo porque lo recontruyo luego con fs

  df["voltaje"] = pd.to_numeric(df["voltaje"], errors="coerce") #convierto a numerico
  df = df.dropna(subset=["voltaje"]).reset_index(drop=True) #elimino datos faltantes y reseto el índice

  df['tiempo']= np.arange(len(df)) / fs #genero el eje temporal con la frecuencia de muestreo

  return df

def filtroPB(fs, ecg):

    f1, f2, orden = 0.5, 40, 3

    # Pasabanda Butterworth
    sos = butter(N=orden, Wn=[f1, f2], btype="bandpass", fs=fs, output="sos")
    ecg_limpio = sosfiltfilt(sos, ecg)

    return ecg_limpio


# FILTRO OPTIMO (WIENER)
# El filtro de Wiener es el que minimiza el error cuadratico medio entre la
# señal filtrada y la señal "deseada". En frecuencia es:
#     H(f) = Sss(f) / (Sss(f) + Snn(f))
# con Sss el espectro de la señal deseada y Snn el del ruido. Donde domina la
# señal H ~ 1 (deja pasar); donde domina el ruido H ~ 0 (atenua).
# Como no conocemos la señal limpia, la estimamos con el latido promedio
# (plantilla) repetido en cada pico R, y el ruido es lo que sobra: ECG - plantilla.


#Latido promedio (mediana de todos los latidos alineados en el pico R).
def plantilla_latido(ecg, picos_R, fs, pre=0.25, post=0.40):
    a, b = int(pre * fs), int(post * fs)
    picos_R = np.asarray(picos_R, dtype=int)
    validos = picos_R[(picos_R >= a) & (picos_R < len(ecg) - b)]
    latidos = np.array([ecg[p - a:p + b] for p in validos])
    plantilla = np.median(latidos, axis=0)

    # Bordes en 0 y suavizados (ventana de Tukey), para que al pegar una
    # plantilla por latido no queden saltos entre un latido y el siguiente.
    plantilla = plantilla - (plantilla[:5].mean() + plantilla[-5:].mean()) / 2
    return plantilla * tukey(len(plantilla), 0.2), a


#Señal "deseada": la plantilla pegada en cada pico R, cero en el resto.
def senal_modelo(n, picos_R, plantilla, a):
    s = np.zeros(n)
    for p in np.asarray(picos_R, dtype=int):
        if p - a >= 0 and p - a + len(plantilla) <= n:
            s[p - a:p - a + len(plantilla)] += plantilla
    return s


#Diseña el Wiener: calcula los espectros de señal y ruido con Welch y arma H(f).
#Usamos tramos largos (4096 muestras) para tener buena resolucion en frecuencias
#bajas y poder separar la deriva de linea de base (< 0,5 Hz) del latido (~1,2 Hz --> en el grupo 3 el intervalo nn típico ronda los 800 ms, que equivale a 1,2 Hz aprox).
def disenar_wiener(ecg, picos_R, fs, nperseg=4096):
    plantilla, a = plantilla_latido(ecg, picos_R, fs)
    s = senal_modelo(len(ecg), picos_R, plantilla, a)
    f, Pss = welch(s, fs, nperseg=nperseg)
    _, Pnn = welch(ecg - s, fs, nperseg=nperseg)
    H = Pss / (Pss + Pnn + 1e-12)   # el 1e-12 evita dividir por cero en 0 Hz
    return f, H, Pss, Pnn


#Aplica el Wiener en frecuencia: FFT, multiplica por H y vuelve al tiempo.
#H es real, asi que no introduce desfasaje (igual que filtfilt).
def filtroWiener(fs, ecg, f, H):
    X = np.fft.rfft(ecg)
    Hx = np.interp(np.fft.rfftfreq(len(ecg), 1 / fs), f, H)
    return np.fft.irfft(X * Hx, n=len(ecg))