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
    """Señal de ECG cruda de un participante, en mV."""
    # usecols evita cargar en memoria las columnas de timestamp y etiqueta,
    # que aca no se usan: son ~2 millones de filas por archivo.
    df = pd.read_csv(CRUDO / codigo / "BitalinoECG.txt", sep="\t", header=None,
                     names=["voltaje", "tiempo", "raw"], usecols=["voltaje"])
    # coerce + dropna descartan cualquier valor no numerico suelto.
    return pd.to_numeric(df["voltaje"], errors="coerce").dropna().to_numpy()


def cargar_picos(codigo):
    """Posiciones de las ondas R ya detectadas por procesar_registros.py."""
    # Se leen de disco en vez de volver a correr Pan-Tompkins sobre la señal.
    return np.load(PROCESADO / "picos" / f"{codigo}.npy")


def fig_crudo_vs_filtrado():
    """Efecto del filtro pasabanda sobre la señal."""
    ecg = cargar_ecg(EJEMPLO)
    filtrado = filtroPB(FS, ecg)

    # Se grafica solo un tramo corto: con 34 minutos de señal no se
    # distinguiria la morfologia de los latidos. El indice de muestra es el
    # segundo multiplicado por la frecuencia de muestreo.
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

    # De todos los picos R del registro, solo los que caen en el tramo que se
    # esta dibujando.
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

    # Mascara de los intervalos que no pasaron los criterios de validez.
    # Se usa para marcarlos en rojo sobre el tacograma RR.
    descartados = tabla.quality_flag != FLAG_OK

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

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

    # subject_groups.csv trae una columna por grupo con los codigos adentro.
    # Se lo da vuelta a un diccionario participante -> grupo para poder
    # colorear cada barra.
    grupos = pd.read_csv(CRUDO / "subject_groups.csv")
    de_grupo = {vp: col for col in grupos.columns for vp in grupos[col].dropna()}
    df["grupo"] = df.participante.map(de_grupo)

    # Ordenado por grupo y, dentro de cada grupo, de mejor a peor calidad.
    df = df.sort_values(["grupo", "pct_descartado"])

    # Mismos colores por grupo que en la figura 12 (scripts/figuras_estadisticas.py).
    colores = {"Group2": "tab:purple", "Group3": "tab:red", "Group4": "tab:green"}

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


def fig_habituacion():
    """Evolucion de la respuesta cardiaca a lo largo de los 16 videos.

    Es el control del principal problema de interpretacion del trabajo: el
    reposo que se usa como referencia esta al final del experimento, asi que
    una diferencia contra el podria deberse al paso del tiempo y no a los
    estimulos. Si fuera asi, deberia verse una tendencia sostenida a lo largo
    de los videos, que es lo que esta figura permite mirar.

    Cada participante se centra en su propio promedio antes de graficarlo.
    Sin centrar, la distancia entre personas (hay quien tiene 70 lpm y quien
    tiene 100) taparia por completo la tendencia, que es lo que interesa.
    """
    tabla = pd.read_csv(RAIZ / "data" / "features" / "metricas.csv")
    videos = tabla[(tabla.condition == "exposicion") &
                   (tabla.window_duration_s == 60) & tabla.valida]

    # Dos metricas: el nivel de pulso y la variabilidad de corto plazo, que
    # son las representantes de dos de las familias de metricas.
    paneles = [("Nmean", "Frecuencia cardíaca [lpm]", "tab:red"),
               ("rMSSD", "rMSSD [ms]", "tab:blue")]

    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, (metrica, etiqueta, color) in zip(ejes, paneles):
        centradas = []

        for _, suyas in videos.groupby("participant_id"):
            suyas = suyas.sort_values("exposure_order")
            datos = suyas[["exposure_order", metrica]].dropna()
            if len(datos) < 8:
                continue

            # Se le resta a cada persona su propio promedio, para que todas
            # arranquen alrededor de cero y se puedan comparar las tendencias.
            y = datos[metrica] - datos[metrica].mean()
            ax.plot(datos.exposure_order, y, color="gray", alpha=0.35,
                    linewidth=0.9)
            centradas.append(pd.Series(y.to_numpy(),
                                       index=datos.exposure_order.to_numpy()))

        # Promedio de todos los participantes en cada posicion de la serie.
        promedio = pd.concat(centradas, axis=1).mean(axis=1).sort_index()
        ax.plot(promedio.index, promedio.to_numpy(), color=color, linewidth=3,
                label="promedio de los 18")

        # Recta ajustada al promedio, para cuantificar la tendencia.
        recta = np.polyfit(promedio.index, promedio.to_numpy(), 1)
        ax.plot(promedio.index, np.polyval(recta, promedio.index), "--",
                color="black", linewidth=1.5,
                label=f"tendencia: {recta[0]:+.2f} por video")

        ax.axhline(0, color="k", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("Orden de exposición al video")
        ax.set_ylabel(f"{etiqueta}\n(centrado en cada participante)")
        ax.set_title(metrica)
        ax.set_xticks(range(1, 17))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Evolución de la respuesta cardíaca a lo largo de los 16 videos\n"
                 "Cada línea gris es un participante", fontsize=12)
    fig.tight_layout()
    return fig, "08_habituacion.png"


def main() -> int:
    """Genera todas las figuras y las guarda en results/figures/."""
    if not (PROCESADO / "picos").exists():
        print("Faltan los picos R. Correr antes: python scripts/procesar_registros.py")
        return 1

    FIGURAS.mkdir(parents=True, exist_ok=True)

    for constructor in (fig_crudo_vs_filtrado, fig_qrs_detectados,
                        fig_tacogramas, fig_falso_positivo,
                        fig_calidad_cohorte, fig_habituacion):
        fig, nombre = constructor()
        fig.savefig(FIGURAS / nombre, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {nombre}")

    print(f"\nFiguras guardadas en {FIGURAS.relative_to(RAIZ)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
