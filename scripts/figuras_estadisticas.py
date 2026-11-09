#Función del archivo: FIGURAS de los resultados estadisticos (P1, P2, P3).

#     QUE HACE : arma las figuras que acompañan a los tests de estadisticas.py
#     ENTRA    : metricas.csv (y los intervalos, para la HR previa al video)
#     SALE     : -09_videos_vs_reposo.png
#                -10_hr_a_lo_largo_de_los_videos.png
#                -11_hr_dentro_del_video.png
#                -12_replicacion_por_grupo.png
#     CORRER   : python scripts/figuras_estadisticas.py
            

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

#Ruta
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from modulos import estadisticas as est
from modulos.ventanas import respuesta_inicial

FIGURAS = RAIZ / "results" / "figures"

# Metricas que se grafican: las representantes de las familias, sin HRV_ihmig (mide tendencia).
METRICAS_GRAFICO = ["Nmean", "SDNN", "rMSSD", "pNN20"]
UNIDADES = {"Nmean": "lpm sobre el reposo", "SDNN": "ms", "rMSSD": "ms", "pNN20": "%"}

COLOR_HR = "tab:red"
COLOR_HRV = "tab:blue"


# Media entre personas y medio ancho del IC 95 % (t de Student), por columna.
# matriz: una fila por persona, una columna por posicion (video o ventana).
def _media_ic(matriz):
    n = matriz.notna().sum()
    media = matriz.mean()
    margen = stats.t.ppf(0.975, n - 1) * matriz.std(ddof=1) / np.sqrt(n)
    return media, margen


# Guarda la figura en results/figures/.
def guardar(fig, nombre):
    FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS / nombre, dpi=150, bbox_inches="tight")


# Figura 09 -- P1: videos vs reposo

# Un panel por metrica. Cada linea gris es una persona: une su valor en el
# reposo con su valor en los videos. La linea gruesa une las medianas.
# Sirve para ver la direccion del cambio persona por persona, que es lo que
# compara el test pareado.
def figura_videos_vs_reposo(metricas):
    resumen = est.resumir_por_participante(metricas)
    p1 = est.p1_exposicion_vs_reposo(metricas).set_index("metrica")

    fig, ejes = plt.subplots(1, len(METRICAS_GRAFICO), figsize=(14, 4))
    for ax, metrica in zip(ejes, METRICAS_GRAFICO):
        videos, reposo = est.pares(resumen, metrica)
        color = COLOR_HR if metrica == "Nmean" else COLOR_HRV

        for r, v in zip(reposo, videos):
            ax.plot([0, 1], [r, v], color="0.75", linewidth=1, marker="o", markersize=3)
        ax.plot([0, 1], [np.median(reposo), np.median(videos)], color=color,
                linewidth=3, marker="o", markersize=8, label="mediana")

        ax.set_xticks([0, 1], ["Reposo", "Videos"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(f"{metrica}\np Holm = {p1.loc[metrica, 'p_holm']:.2f}")
        ax.set_ylabel(UNIDADES[metrica])
        ax.grid(alpha=0.3, axis="y")

    ejes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("P1. Videos vs reposo: cada línea es una persona (Group3, n = 18)")
    fig.tight_layout()
    return fig


# Figura 10 -- P3a: HR y HRV a lo largo de los 16 videos

# Media ± IC 95 % entre personas en cada posicion de video (1 a 16), con el
# nivel del reposo como referencia. La linea vertical marca el cambio de
# mitad: las posiciones 1-8 siempre fueron los clips 1-8, y las 9-16 los 9-16.
def figura_a_lo_largo_de_los_videos(metricas, lista_metricas=("Nmean", "rMSSD")):
    videos = metricas[(metricas.window_duration_s == 60)
                      & (metricas.condition == "exposicion") & metricas.valida]
    reposo = est.resumir_por_participante(metricas)
    reposo = reposo[reposo.condition == "reposo"]

    fig, ejes = plt.subplots(len(lista_metricas), 1, figsize=(9, 3.5 * len(lista_metricas)),
                             sharex=True)
    for ax, metrica in zip(np.atleast_1d(ejes), lista_metricas):
        matriz = videos.pivot_table(index="participant_id", columns="exposure_order",
                                    values=metrica)
        media, margen = _media_ic(matriz)
        color = COLOR_HR if metrica == "Nmean" else COLOR_HRV

        ax.plot(media.index, media, color=color, marker="o", label="videos (media ± IC 95 %)")
        ax.fill_between(media.index, media - margen, media + margen, color=color, alpha=0.2)
        ax.axhline(reposo[metrica].mean(), color="k", linestyle="--", linewidth=1,
                   label="reposo (media)")
        ax.axvline(8.5, color="0.5", linestyle=":", linewidth=1.5)
        ax.text(8.6, ax.get_ylim()[1], " cambio de mitad", va="top", fontsize=8, color="0.4")

        ax.set_ylabel(f"{metrica} [{UNIDADES[metrica]}]")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    np.atleast_1d(ejes)[-1].set_xlabel("Posición del video (en el orden en que cada persona los vio)")
    np.atleast_1d(ejes)[-1].set_xticks(range(1, 17))
    fig.suptitle("P3a. HR y HRV a lo largo de la sesión")
    fig.tight_layout()
    return fig


# Figura 11 -- P2 y P3b: HR dentro del video

# HR (Nmean) en los 10 s previos al video y en sus 6 ventanas de 10 s.
# Para cada persona se promedian sus 16 videos; despues, media ± IC 95 %
# entre personas. Muestra a la vez:
#   - P2: no hay salto al empezar el video (la HR ya estaba alta antes);
#   - P3b: la HR baja en los primeros ~20 s y queda en el nivel del reposo.
def figura_hr_dentro_del_video(metricas):
    cortas = metricas[(metricas.window_duration_s == 10)
                      & (metricas.condition == "exposicion")].copy()
    por_video = cortas.groupby(["participant_id", "event_id"]).window_start_s
    cortas["posicion"] = por_video.rank(method="first").astype(int) - 1
    cortas = cortas[cortas.valida]

    # HR de los 10 s previos, en la misma escala que Nmean:
    # Nmean(primera ventana) - (HR primeros 10 s - HR 10 s previos).
    previo = respuesta_inicial()[["participant_id", "event_id", "respuesta_lpm"]]
    primera = cortas[cortas.posicion == 0][["participant_id", "event_id", "Nmean"]]
    previo = primera.merge(previo, on=["participant_id", "event_id"])
    previo["Nmean"] = previo["Nmean"] - previo["respuesta_lpm"]
    previo["posicion"] = -1

    columnas = ["participant_id", "posicion", "Nmean"]
    juntas = pd.concat([previo[columnas], cortas[columnas]])

    # Una fila por persona, una columna por tramo (promedio de sus 16 videos).
    matriz = juntas.pivot_table(index="participant_id", columns="posicion", values="Nmean")
    media, margen = _media_ic(matriz)
    centros = np.array(media.index, dtype=float) * 10 + 5     # -1 -> -5 s, 0 -> 5 s, ...

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axvspan(0, 60, color="0.93", label="video")
    ax.plot(centros, media, color=COLOR_HR, marker="o", label="HR (media ± IC 95 %)")
    ax.fill_between(centros, media - margen, media + margen, color=COLOR_HR, alpha=0.2)
    ax.axhline(0, color="k", linestyle="--", linewidth=1, label="nivel del reposo")
    ax.axvline(0, color="0.4", linewidth=1)

    ax.set_xticks([-5, 5, 15, 25, 35, 45, 55],
                  ["10 s\nprevios", "0-10", "10-20", "20-30", "30-40", "40-50", "50-60"])
    ax.set_xlabel("Tramo de 10 s (segundos desde el inicio del video)")
    ax.set_ylabel("Nmean [lpm sobre el reposo]")
    ax.set_title("P2 y P3b. HR antes y durante el video (promedio de los 16 videos)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# Figura 12 -- Replicacion de P3b en los tres grupos

# HR (Nmean) en las 6 ventanas de 10 s del video, una curva por grupo.
# A cada persona se le resta su primera ventana: asi todas las curvas
# arrancan en 0 y se ve solo cuanto baja la HR dentro del video. Los niveles
# no se comparan entre grupos, porque Group2 y Group4 tienen biofeedback.
def figura_replicacion_por_grupo():
    colores = {"Group3": COLOR_HR, "Group2": "tab:purple", "Group4": "tab:green"}

    fig, ax = plt.subplots(figsize=(9, 4))
    for grupo in est.GRUPOS:
        metricas = est.metricas_de_grupo(grupo)
        cortas = metricas[(metricas.window_duration_s == 10)
                          & (metricas.condition == "exposicion")].copy()
        por_video = cortas.groupby(["participant_id", "event_id"]).window_start_s
        cortas["posicion"] = por_video.rank(method="first").astype(int) - 1
        cortas = cortas[cortas.valida]

        # Una fila por persona, una columna por ventana (promedio de sus 16 videos).
        matriz = cortas.pivot_table(index="participant_id", columns="posicion",
                                    values="Nmean")
        matriz = matriz.sub(matriz[0], axis=0).dropna(subset=[0])
        media, margen = _media_ic(matriz)
        centros = np.array(media.index, dtype=float) * 10 + 5

        ax.plot(centros, media, color=colores[grupo], marker="o",
                label=f"{grupo} (n = {len(matriz)})")
        ax.fill_between(centros, media - margen, media + margen,
                        color=colores[grupo], alpha=0.15)

    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xticks([5, 15, 25, 35, 45, 55],
                  ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60"])
    ax.set_xlabel("Tramo de 10 s (segundos desde el inicio del video)")
    ax.set_ylabel("Cambio de HR respecto de los\nprimeros 10 s [lpm]")
    ax.set_title("P3b en los tres grupos: la HR baja dentro del video (media ± IC 95 %)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# Al ejecutar: genera y guarda las cuatro figuras

if __name__ == "__main__":
    metricas = est.cargar_metricas()
    figuras = {
        "09_videos_vs_reposo.png": figura_videos_vs_reposo(metricas),
        "10_hr_a_lo_largo_de_los_videos.png": figura_a_lo_largo_de_los_videos(metricas),
        "11_hr_dentro_del_video.png": figura_hr_dentro_del_video(metricas),
        "12_replicacion_por_grupo.png": figura_replicacion_por_grupo(),
    }
    for nombre, fig in figuras.items():
        guardar(fig, nombre)
        print(f"  {nombre}")
    print("\nFiguras guardadas en results/figures/")
