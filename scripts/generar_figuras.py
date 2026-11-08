"""generar_figuras.py -- ARMA LAS FIGURAS del informe y las guarda como PNG.

    QUE HACE : genera las figuras que documentan el procesamiento de la señal.
    ENTRA    : data/raw/ y lo que dejo procesar_registros.py en data/processed/
    SALE     : results/figures/*.png
    CORRER   : python scripts/generar_figuras.py

Hay que haber corrido antes scripts/procesar_registros.py, porque las figuras
usan los picos R ya detectados.
"""

import sys
from pathlib import Path

import matplotlib

# Backend sin ventana: el script guarda archivos, no abre graficos en pantalla.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from modulos.intervalos import FLAG_OK, construir_intervalos
from modulos.preprocesamiento import filtroPB

FS = 100
CRUDO = RAIZ / "data" / "raw"
PROCESADO = RAIZ / "data" / "processed"
FIGURAS = RAIZ / "results" / "figures"

# Participante de referencia para las figuras de ejemplo. VP03 tiene cero
# intervalos descartados, asi que sirve para mostrar el caso limpio.
EJEMPLO = "VP03"
EJEMPLO_INICIO, EJEMPLO_FIN = 600, 610     # tramo de 10 s a graficar

# VP12 tiene un episodio claro de deteccion duplicada alrededor del segundo 237.
EJEMPLO_ARTEFACTO = "VP12"
ARTEFACTO_INICIO, ARTEFACTO_FIN = 234, 242


def cargar_ecg(codigo):
    df = pd.read_csv(CRUDO / codigo / "BitalinoECG.txt", sep="\t", header=None,
                     names=["voltaje", "tiempo", "raw"], usecols=["voltaje"])
    return pd.to_numeric(df["voltaje"], errors="coerce").dropna().to_numpy()


def cargar_picos(codigo):
    return np.load(PROCESADO / "picos" / f"{codigo}.npy")


def fig_crudo_vs_filtrado():
    """Efecto del filtro pasabanda sobre la señal."""
    ecg = cargar_ecg(EJEMPLO)
    filtrado = filtroPB(FS, ecg)

    i0, i1 = EJEMPLO_INICIO * FS, EJEMPLO_FIN * FS
    t = np.arange(i0, i1) / FS

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax1.plot(t, ecg[i0:i1], color="tab:gray", linewidth=1)
    ax1.set_title(f"{EJEMPLO} — ECG crudo")
    ax1.set_ylabel("Amplitud [mV]")

    ax2.plot(t, filtrado[i0:i1], color="tab:blue", linewidth=1)
    ax2.set_title("ECG filtrado (pasabanda Butterworth 0,5–40 Hz, fase cero)")
    ax2.set_ylabel("Amplitud [mV]")
    ax2.set_xlabel("Tiempo [s]")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, "01_crudo_vs_filtrado.png"


def fig_qrs_detectados():
    """Complejos QRS detectados sobre la señal filtrada."""
    filtrado = filtroPB(FS, cargar_ecg(EJEMPLO))
    picos = cargar_picos(EJEMPLO)

    i0, i1 = EJEMPLO_INICIO * FS, EJEMPLO_FIN * FS
    t = np.arange(i0, i1) / FS
    en_tramo = picos[(picos >= i0) & (picos < i1)]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, filtrado[i0:i1], color="tab:blue", linewidth=1)
    ax.plot(en_tramo / FS, filtrado[en_tramo], "o", color="crimson",
            markersize=7, label=f"R detectadas ({len(en_tramo)})")
    ax.set_title(f"{EJEMPLO} — complejos QRS detectados con Pan-Tompkins")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud [mV]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, "02_qrs_detectados.png"


def fig_tacogramas():
    """Tacograma RR frente a tacograma NN, marcando los intervalos descartados."""
    tabla = construir_intervalos(cargar_picos(EJEMPLO_ARTEFACTO), FS,
                                 EJEMPLO_ARTEFACTO)
    descartados = tabla.quality_flag != FLAG_OK

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, sharey=True)

    ax1.plot(tabla.rr_end_s, tabla.rr_ms, color="tab:gray", linewidth=0.8)
    ax1.plot(tabla.rr_end_s[descartados], tabla.rr_ms[descartados], "o",
             color="crimson", markersize=3,
             label=f"descartados ({descartados.sum()})")
    ax1.set_title(f"{EJEMPLO_ARTEFACTO} — tacograma RR (serie completa)")
    ax1.set_ylabel("RR [ms]")
    ax1.legend(loc="upper right")

    ax2.plot(tabla.rr_end_s, tabla.nn_ms, color="tab:blue", linewidth=0.8)
    ax2.set_title("Tacograma NN (tras descartar los intervalos no válidos)")
    ax2.set_ylabel("NN [ms]")
    ax2.set_xlabel("Tiempo [s]")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, "03_tacogramas_rr_nn.png"


def fig_falso_positivo():
    """Detalle de un artefacto detectado como latido.

    Alrededor del segundo 237 hay una deflexión negativa de gran amplitud que
    no corresponde a un complejo QRS. El detector la marca igual, porque el
    refinamiento de la posición de R busca el máximo del valor ABSOLUTO de la
    señal filtrada, y esa excursión supera en módulo al pico R verdadero.
    El intervalo real de unos 750 ms queda partido en dos de 490 y 260 ms,
    que es lo que los criterios de validez detectan y descartan.
    """
    filtrado = filtroPB(FS, cargar_ecg(EJEMPLO_ARTEFACTO))
    picos = cargar_picos(EJEMPLO_ARTEFACTO)
    tabla = construir_intervalos(picos, FS, EJEMPLO_ARTEFACTO)

    i0, i1 = ARTEFACTO_INICIO * FS, ARTEFACTO_FIN * FS
    t = np.arange(i0, i1) / FS
    en_tramo = picos[(picos >= i0) & (picos < i1)]

    # Intervalos del tramo, separados segun hayan pasado o no los criterios.
    tramo = tabla[(tabla.rr_end_s >= ARTEFACTO_INICIO) &
                  (tabla.rr_end_s <= ARTEFACTO_FIN)]
    malos = tramo[tramo.quality_flag != FLAG_OK]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6),
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(t, filtrado[i0:i1], color="tab:blue", linewidth=1)
    ax1.plot(en_tramo / FS, filtrado[en_tramo], "o", color="crimson",
             markersize=7, label="R detectadas")
    ax1.set_title(f"{EJEMPLO_ARTEFACTO} — artefacto detectado como latido: "
                  "parte un intervalo de ~750 ms en dos")
    ax1.set_ylabel("Amplitud [mV]")
    ax1.legend(loc="upper right")

    ax2.plot(tramo.rr_end_s, tramo.rr_ms, "o-", color="tab:gray",
             markersize=5, linewidth=1, label="RR")
    ax2.plot(malos.rr_end_s, malos.rr_ms, "o", color="crimson", markersize=8,
             label="descartados")
    # Mediana del participante, como referencia de cuanto deberia durar un latido.
    ax2.axhline(tabla.nn_ms.median(), color="k", linestyle="--", linewidth=1,
                label=f"NN mediana = {tabla.nn_ms.median():.0f} ms")
    ax2.set_ylabel("RR [ms]")
    ax2.set_xlabel("Tiempo [s]")
    ax2.legend(loc="lower right", fontsize=8)

    for ax in (ax1, ax2):
        ax.set_xlim(ARTEFACTO_INICIO, ARTEFACTO_FIN)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, "04_falso_positivo.png"


def fig_calidad_cohorte():
    """Porcentaje de intervalos descartados en cada participante, por grupo."""
    df = pd.read_csv(PROCESADO / "calidad_deteccion.csv")
    grupos = pd.read_csv(CRUDO / "subject_groups.csv")
    de_grupo = {vp: col for col in grupos.columns for vp in grupos[col].dropna()}
    df["grupo"] = df.participante.map(de_grupo)
    df = df.sort_values(["grupo", "pct_descartado"])

    colores = {"Group2": "tab:orange", "Group3": "tab:green", "Group4": "tab:red"}

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df.participante, df.pct_descartado,
           color=[colores.get(g, "tab:gray") for g in df.grupo])
    ax.axhline(5, color="k", linestyle="--", linewidth=1,
               label="umbral de revisión (5 %)")

    ax.set_ylabel("Intervalos descartados [%]")
    # La escala logaritmica hace falta porque VP70 descarta el 60 % y el resto
    # menos del 10 %. Su contrapartida es que los participantes con 0 % no
    # dibujan barra, asi que se aclara en el titulo para no leerlos como
    # datos faltantes.
    ax.set_yscale("log")
    sin_descartes = (df.pct_descartado == 0).sum()
    ax.set_title("Calidad de la detección por participante\n"
                 f"(escala logarítmica; {sin_descartes} participantes sin "
                 "ningún intervalo descartado no dibujan barra)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)

    manual = [plt.Line2D([0], [0], color=c, lw=6, label=g)
              for g, c in colores.items()]
    ax.legend(handles=manual + ax.get_legend_handles_labels()[0])
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig, "05_calidad_cohorte.png"


def main() -> int:
    if not (PROCESADO / "picos").exists():
        print("Faltan los picos R. Correr antes: python scripts/procesar_registros.py")
        return 1

    FIGURAS.mkdir(parents=True, exist_ok=True)

    for constructor in (fig_crudo_vs_filtrado, fig_qrs_detectados,
                        fig_tacogramas, fig_falso_positivo,
                        fig_calidad_cohorte):
        fig, nombre = constructor()
        fig.savefig(FIGURAS / nombre, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {nombre}")

    print(f"\nFiguras guardadas en {FIGURAS.relative_to(RAIZ)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
