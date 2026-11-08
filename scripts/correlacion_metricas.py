"""correlacion_metricas.py -- CUANTAS cosas distintas miden las metricas.

    QUE HACE : mide cuanto se solapan las metricas entre si y las agrupa en
               familias, para saber cuantas comparaciones independientes se
               estan haciendo en realidad.
    ENTRA    : data/features/metricas.csv
    SALE     : results/tables/correlacion_metricas.csv
               results/tables/familias_metricas.csv
               results/figures/07_correlacion_metricas.png
    CORRER   : python scripts/correlacion_metricas.py

--------------------------------------------------------------------------

Por que hace falta
------------------
Las catorce metricas no son catorce cosas distintas. SD1 y rMSSD son casi la
misma formula, CVNN es SDNN dividido avNN, y pNN50 y pNN20 cuentan lo mismo
con distinto umbral. Comparar las catorce entre condiciones y despues corregir
por catorce comparaciones independientes seria castigarse de mas, porque las
comparaciones no son independientes.

Agruparlas en familias permite elegir una representante de cada una y reducir
el numero de pruebas a las que realmente aportan informacion distinta.

Como se calcula
---------------
La correlacion se calcula DENTRO de cada participante, sobre sus propias
ventanas, y recien despues se promedia entre participantes.

Mezclar las ventanas de todos en una sola matriz daria correlaciones infladas:
quedarian dominadas por las diferencias entre personas (quien tiene mucha
variabilidad la tiene alta en todas las metricas a la vez), y no por como se
mueven las metricas dentro de una misma persona, que es el nivel en el que se
hacen las comparaciones del trabajo.

El promedio se hace con la transformacion z de Fisher. Los coeficientes de
correlacion no se promedian bien en forma directa porque su distribucion se
deforma cerca de 1; la transformacion z = arctanh(r) la simetriza, se promedia
ahi y se vuelve con la tangente hiperbolica.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from modulos.metricas import METRICAS

CARACTERISTICAS = RAIZ / "data" / "features"
FIGURAS = RAIZ / "results" / "figures"
TABLAS = RAIZ / "results" / "tables"

DURACION = 60          # se usa el analisis principal, de ventanas de 60 s
MINIMO_VENTANAS = 10   # participantes con menos ventanas no aportan correlacion

# Dos metricas se consideran de la misma familia si su correlacion absoluta
# media supera este valor. Es un umbral convencional para "redundante".
UMBRAL_FAMILIA = 0.80


def matriz_promedio(df, columnas):
    """Correlacion media entre metricas, calculada dentro de cada participante.

    Devuelve la matriz promediada y la cantidad de participantes que aportaron.
    """
    acumulador = []

    for _, ventanas in df.groupby("participant_id"):
        if len(ventanas) < MINIMO_VENTANAS:
            continue
        r = ventanas[columnas].corr(method="pearson")

        # Transformacion z de Fisher. Se recortan los valores exactamente en
        # +-1 porque arctanh diverge ahi; pasa cuando dos metricas son
        # linealmente dependientes dentro de un participante.
        acumulador.append(np.arctanh(np.clip(r.to_numpy(), -0.9999, 0.9999)))

    if not acumulador:
        raise SystemExit("Ningun participante tiene suficientes ventanas.")

    promedio_z = np.nanmean(np.stack(acumulador), axis=0)
    promedio_r = np.tanh(promedio_z)
    np.fill_diagonal(promedio_r, 1.0)

    return pd.DataFrame(promedio_r, index=columnas, columns=columnas), len(acumulador)


def agrupar(matriz):
    """Agrupa las metricas en familias por su correlacion absoluta.

    Se usa 1 - |r| como distancia: dos metricas que se mueven juntas, en el
    mismo sentido o en el contrario, aportan la misma informacion.
    """
    distancia = 1 - matriz.abs().to_numpy()
    np.fill_diagonal(distancia, 0.0)
    # squareform exige simetria exacta; el promedio puede dejar diferencias
    # minimas de redondeo.
    distancia = (distancia + distancia.T) / 2

    enlace = linkage(squareform(distancia, checks=False), method="average")
    etiquetas = fcluster(enlace, t=1 - UMBRAL_FAMILIA, criterion="distance")
    orden = leaves_list(enlace)

    return etiquetas, [matriz.columns[i] for i in orden]


def graficar(matriz, orden, familias):
    """Mapa de calor de la matriz, con las metricas ordenadas por familia."""
    m = matriz.loc[orden, orden]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    imagen = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(orden)))
    ax.set_yticks(range(len(orden)))
    ax.set_xticklabels(orden, rotation=90)
    ax.set_yticklabels(orden)

    # Se escribe el valor en cada celda; el color del texto se elige segun la
    # intensidad del fondo para que siempre se lea.
    for i in range(len(orden)):
        for j in range(len(orden)):
            valor = m.iloc[i, j]
            ax.text(j, i, f"{valor:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(valor) > 0.6 else "black")

    # Lineas que separan las familias, siguiendo el orden del agrupamiento.
    familia_por_metrica = dict(zip(matriz.columns, familias))
    for i in range(1, len(orden)):
        if familia_por_metrica[orden[i]] != familia_por_metrica[orden[i - 1]]:
            ax.axhline(i - 0.5, color="k", linewidth=2)
            ax.axvline(i - 0.5, color="k", linewidth=2)

    fig.colorbar(imagen, ax=ax, label="correlación media dentro de participante")
    ax.set_title("Solapamiento entre métricas\n"
                 f"(agrupadas por |r| > {UMBRAL_FAMILIA}; "
                 "calculada por participante y promediada con z de Fisher)")
    fig.tight_layout()
    return fig


def main() -> int:
    archivo = CARACTERISTICAS / "metricas.csv"
    if not archivo.exists():
        print("Falta metricas.csv. Correr antes: python modulos/ventanas.py")
        return 1

    df = pd.read_csv(archivo)
    df = df[df.valida & (df.window_duration_s == DURACION)]

    # Se excluyen las metricas sin variacion util para este analisis.
    columnas = [m for m in METRICAS if df[m].notna().sum() > 0
                and df[m].std() > 0]

    matriz, n_participantes = matriz_promedio(df, columnas)
    familias, orden = agrupar(matriz)

    print(f"Participantes            : {n_participantes}")
    print(f"Ventanas de {DURACION} s por persona : "
          f"{df.groupby('participant_id').size().median():.0f} (mediana)\n")

    tabla_familias = (pd.DataFrame({"metrica": matriz.columns,
                                    "familia": familias})
                      .sort_values(["familia", "metrica"]))

    print("=== familias ===")
    for numero, grupo in tabla_familias.groupby("familia"):
        print(f"  Familia {numero}: {', '.join(grupo.metrica)}")

    print(f"\nDe {len(columnas)} metricas quedan "
          f"{tabla_familias.familia.nunique()} grupos independientes.")

    print("\n=== pares mas redundantes ===")
    pares = (matriz.where(np.triu(np.ones(matriz.shape), 1).astype(bool))
             .stack().abs().sort_values(ascending=False).head(6))
    for (a, b), r in pares.items():
        print(f"  {a:<10} {b:<10} |r| = {r:.3f}")

    TABLAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    matriz.round(3).to_csv(TABLAS / "correlacion_metricas.csv")
    tabla_familias.to_csv(TABLAS / "familias_metricas.csv", index=False)
    graficar(matriz, orden, familias).savefig(
        FIGURAS / "07_correlacion_metricas.png", dpi=150, bbox_inches="tight")

    print("\nTabla  : results/tables/correlacion_metricas.csv")
    print("Familias: results/tables/familias_metricas.csv")
    print("Figura : results/figures/07_correlacion_metricas.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
