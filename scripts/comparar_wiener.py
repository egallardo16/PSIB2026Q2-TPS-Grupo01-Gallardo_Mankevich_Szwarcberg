# Función del archivo: compara el filtro optimo (Wiener) con el Butterworth que usa el pipeline.

# en cada participante de Group3 filtra el ECG crudo con los dos
# filtros, detecta los QRS sobre cada uno y compara cuanto limpian la señal y
# si cambia la deteccion.
# ENTRA: data/raw/VPxx/BitalinoECG.txt y los picos R de data/processed/picos/
# SALE:  results/tables/wiener_por_participante.csv  (una fila por persona)
#        results/tables/wiener_resumen.csv           (Butterworth vs Wiener, con Wilcoxon)
#        results/figures/13_wiener_vs_butterworth.png
# CORRER: python scripts/comparar_wiener.py

# Se usa para comparar sólamente porque el pipeline sigue usando el Butterworth.

# Como se compara:
# - SNR [dB]: potencia de la plantilla de latido / potencia de lo que sobra.
# - Deteccion: latidos detectados, coincidencia de picos R entre los dos
#   filtros y % de intervalos descartados por los criterios de la serie NN.
#   Esta es la comparacion que importa: si el Wiener no mejora la deteccion,
#   no hay motivo para cambiar el filtro del pipeline.

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfreqz

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from modulos import estadisticas as est
from modulos.intervalos import construir_intervalos, resumen_calidad
from modulos.preprocesamiento import (disenar_wiener, filtroPB, filtroWiener,
                                      plantilla_latido, senal_modelo)
from modulos.procesamiento import detectar_qrs, panTompkins
from modulos.ventanas import cohorte_group3
from scripts.procesar_registros import cargar_ecg

FS = 100
PICOS = RAIZ / "data" / "processed" / "picos"
TABLAS = RAIZ / "results" / "tables"
FIGURAS = RAIZ / "results" / "figures"

EJEMPLO = "VP03"              # mismo participante y tramo que las figuras 01 y 02
EJEMPLO_INICIO, EJEMPLO_FIN = 600, 610
TOLERANCIA_S = 0.05           # dos picos R "coinciden" si estan a menos de 50 ms


#SNR de una señal: plantilla de latido vs lo que sobra, con los picos de referencia.
def snr_db(x, picos_R):
    plantilla, a = plantilla_latido(x, picos_R, FS)
    s = senal_modelo(len(x), picos_R, plantilla, a)
    return 10 * np.log10(np.var(s) / np.var(x - s))


#Porcentaje de picos de "a" que tienen un pico de "b" a menos de TOLERANCIA_S.
def coincidencia(a, b):
    b = np.sort(b)
    pos = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    distancia = np.minimum(np.abs(a - b[pos - 1]), np.abs(a - b[pos])) / FS
    return 100 * np.mean(distancia <= TOLERANCIA_S)


#Filtra un participante con los dos filtros y devuelve las medidas de la comparacion.
def comparar(codigo):
    crudo = cargar_ecg(codigo)
    picos_ref = np.load(PICOS / f"{codigo}.npy")    # los del pipeline (Butterworth)

    butter_ecg = filtroPB(FS, crudo)
    f, H, _, _ = disenar_wiener(crudo, picos_ref, FS)
    wiener_ecg = filtroWiener(FS, crudo, f, H)

    # Deteccion de QRS sobre la señal filtrada con Wiener (igual que el pipeline).
    integrado, filtrado = panTompkins(FS, wiener_ecg)
    picos_wiener = detectar_qrs(FS, integrado, filtrado)

    calidad_b = resumen_calidad(construir_intervalos(picos_ref, FS, codigo))
    calidad_w = resumen_calidad(construir_intervalos(picos_wiener, FS, codigo))

    return {
        "participant_id": codigo,
        "snr_crudo_db": snr_db(crudo, picos_ref),
        "snr_butterworth_db": snr_db(butter_ecg, picos_ref),
        "snr_wiener_db": snr_db(wiener_ecg, picos_ref),
        "latidos_butterworth": len(picos_ref),
        "latidos_wiener": len(picos_wiener),
        "coincidencia_pct": coincidencia(picos_wiener, picos_ref),
        "descartado_butterworth_pct": calidad_b["pct_descartado"],
        "descartado_wiener_pct": calidad_w["pct_descartado"],
    }


#Wilcoxon pareado (el mismo test del resto del trabajo) para cada medida.
def resumir(tabla):
    filas = []
    for medida, col_b, col_w in [
            ("SNR [dB]", "snr_butterworth_db", "snr_wiener_db"),
            ("Latidos detectados", "latidos_butterworth", "latidos_wiener"),
            ("Intervalos descartados [%]", "descartado_butterworth_pct", "descartado_wiener_pct")]:
        fila = {"medida": medida,
                "mediana_butterworth": tabla[col_b].median(),
                "mediana_wiener": tabla[col_w].median()}
        # Si las dos columnas son identicas no hay nada que testear.
        if (tabla[col_w] - tabla[col_b]).abs().sum() == 0:
            fila.update({"n": len(tabla), "diferencia": 0.0, "p": np.nan})
        else:
            fila.update(est.test_w_pareado(tabla[col_w], tabla[col_b]))
        filas.append(fila)
    return pd.DataFrame(filas)


#Figura: arriba, el diseño del Wiener en el ejemplo; abajo, el ECG con cada filtro.
def figura():
    crudo = cargar_ecg(EJEMPLO)
    picos = np.load(PICOS / f"{EJEMPLO}.npy")
    f, H, Pss, Pnn = disenar_wiener(crudo, picos, FS)
    wiener_ecg = filtroWiener(FS, crudo, f, H)
    butter_ecg = filtroPB(FS, crudo)

    # Respuesta del Butterworth: |H|^2 porque filtfilt lo aplica dos veces.
    sos = butter(3, [0.5, 40], btype="bandpass", fs=FS, output="sos")
    fb, hb = sosfreqz(sos, worN=2048, fs=FS)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

    ax1.semilogy(f, Pss, color="tab:blue", label="Sss: señal (plantilla)")
    ax1.semilogy(f, Pnn, color="tab:gray", label="Snn: ruido (ECG - plantilla)")
    ax1.set_xlabel("Frecuencia [Hz]")
    ax1.set_ylabel("PSD [mV²/Hz]")
    ax1.set_xlim(0, 50)
    ax1.grid(alpha=0.3)
    gan = ax1.twinx()
    gan.plot(f, H, color="tab:red", label="H Wiener = Sss / (Sss + Snn)")
    gan.plot(fb, np.abs(hb) ** 2, color="k", linestyle="--", label="Butterworth 0,5-40 Hz")
    gan.set_ylabel("Ganancia")
    gan.set_ylim(0, 1.05)
    lineas = ax1.get_legend_handles_labels()[0] + gan.get_legend_handles_labels()[0]
    ax1.legend(lineas, [l.get_label() for l in lineas], loc="upper right", fontsize=8)
    ax1.set_title(f"{EJEMPLO} — diseño del filtro de Wiener")

    i0, i1 = EJEMPLO_INICIO * FS, EJEMPLO_FIN * FS
    t = np.arange(i0, i1) / FS
    ax2.plot(t, crudo[i0:i1] + 3, color="tab:gray", linewidth=1, label="crudo (+3 mV)")
    ax2.plot(t, butter_ecg[i0:i1] + 1.5, color="k", linewidth=1, label="Butterworth (+1,5 mV)")
    ax2.plot(t, wiener_ecg[i0:i1], color="tab:red", linewidth=1, label="Wiener")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Amplitud [mV]")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3)
    ax2.set_title("ECG con cada filtro (desplazados para verlos)")

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    tabla = pd.DataFrame([comparar(c) for c in cohorte_group3()]).round(2)
    resumen = resumir(tabla)

    TABLAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(TABLAS / "wiener_por_participante.csv", index=False)
    resumen.to_csv(TABLAS / "wiener_resumen.csv", index=False)
    figura().savefig(FIGURAS / "13_wiener_vs_butterworth.png", dpi=150, bbox_inches="tight")

    pd.set_option("display.width", 200)
    print(tabla.to_string(index=False))
    print()
    print(resumen.round(3).to_string(index=False))
    print("\nGuardado en results/tables/wiener_*.csv y results/figures/13_wiener_vs_butterworth.png")
