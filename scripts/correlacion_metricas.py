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

# Orden de preferencia para elegir la representante de cada familia.
#
# Las familias salen de los datos, pero cual de sus miembros se reporta es una
# decision nuestra, y se toma con criterios fijados de antemano para que no
# dependa de que resultado dio cada una. Los criterios, en orden:
#
#   1. Que sea la metrica estandar en la bibliografia de HRV. Si se reporta
#      SD1 en vez de rMSSD, el resultado no se puede comparar con otros
#      trabajos; rMSSD y SDNN tienen incluso valores de referencia publicados
#      (Shaffer y Ginsberg, 2017).
#   2. Que tenga una interpretacion fisiologica clara. rMSSD refleja actividad
#      parasimpatica; NFD y NSD son definiciones propias del articulo que no
#      significan nada fuera de el.
#   3. Que sea una proporcion o un promedio antes que un conteo absoluto. Los
#      conteos como NN50 dependen de cuantos latidos entren en la ventana.
#
# Dentro de cada familia las metricas correlacionan por encima de 0.80 (y en
# varios casos por encima de 0.99), asi que la eleccion casi no cambia el
# resultado del analisis. Lo que importa no es cual se elige, sino no contar
# varias versiones de la misma cantidad como si fueran hallazgos distintos.
PRIORIDAD = [
    "rMSSD",      # estandar de variabilidad de corto plazo; ademas quedo
                  # entre las estables en ventanas de 10 s
    "SDNN",       # estandar de variabilidad total, con normas publicadas
    "Nmean",      # nivel de pulso, normalizado por el reposo de cada persona
    "avNN",       # nivel de pulso sin normalizar
    "pNN50",      # estandar, pero es una proporcion sobre pocos casos
    "pNN20",      # igual que pNN50 con un umbral menos convencional
    "CVNN",       # SDNN relativizado por avNN
    "SD1",        # Poincare; equivalente a rMSSD salvo una constante
    "SD2",        # Poincare; equivalente a SDNN salvo una constante
    "std",        # dispersion de la frecuencia instantanea
    "NFD",        # definicion propia del articulo
    "NSD",        # definicion propia del articulo
    "NN50",       # conteo absoluto: depende del largo de la ventana
    "HRV_ihmig",  # ultima: su formula telescopa (ver modulos/metricas.py)
]


def matriz_promedio(df, columnas):
    """Correlacion media entre metricas, calculada dentro de cada participante.

    Devuelve la matriz promediada y la cantidad de participantes que aportaron.
    """
    acumulador = []

    # Una matriz por participante: se usan solo SUS ventanas, de modo que la
    # correlacion refleje como se mueven las metricas dentro de esa persona.
    for _, ventanas in df.groupby("participant_id"):
        # Con pocas ventanas la correlacion es inestable y ensuciaria el
        # promedio, asi que ese participante no se usa.
        if len(ventanas) < MINIMO_VENTANAS:
            continue

        # Matriz de correlacion de Pearson entre todas las metricas, usando
        # las ventanas de este participante como observaciones.
        r = ventanas[columnas].corr(method="pearson")

        # Transformacion z de Fisher antes de promediar. Se recortan los
        # valores exactamente en +-1 porque arctanh diverge ahi; eso pasa
        # cuando dos metricas son linealmente dependientes, como rMSSD y SD1.
        acumulador.append(np.arctanh(np.clip(r.to_numpy(), -0.9999, 0.9999)))

    if not acumulador:
        raise SystemExit("Ningun participante tiene suficientes ventanas.")

    # Se apilan las matrices de todos los participantes y se promedian celda
    # por celda, todavia en el espacio z.
    promedio_z = np.nanmean(np.stack(acumulador), axis=0)

    # Vuelta al espacio de correlaciones con la tangente hiperbolica, que es
    # la inversa de arctanh.
    promedio_r = np.tanh(promedio_z)

    # El recorte de +-0.9999 deja la diagonal en 0.9999 en vez de 1; se
    # restituye para que la matriz se lea bien.
    np.fill_diagonal(promedio_r, 1.0)

    return pd.DataFrame(promedio_r, index=columnas, columns=columnas), len(acumulador)


def agrupar(matriz):
    """Agrupa las metricas en familias por su correlacion absoluta.

    Se usa 1 - |r| como distancia: dos metricas que se mueven juntas, en el
    mismo sentido o en el contrario, aportan la misma informacion.
    """
    # Se pasa de correlacion a distancia. Se toma el valor absoluto porque dos
    # metricas que se mueven en sentidos opuestos (r = -1) son igual de
    # redundantes que dos que se mueven juntas (r = +1): una se deduce de la
    # otra. Asi, |r| = 1 da distancia 0 y |r| = 0 da distancia 1.
    distancia = 1 - matriz.abs().to_numpy()

    # La distancia de una metrica consigo misma es cero por definicion.
    np.fill_diagonal(distancia, 0.0)

    # squareform exige simetria exacta, y el promedio entre participantes
    # puede dejar diferencias minimas de redondeo entre la celda (i,j) y la
    # (j,i). Se promedian para forzar la simetria.
    distancia = (distancia + distancia.T) / 2

    # Agrupamiento jerarquico. El metodo "average" fusiona grupos segun la
    # distancia media entre sus miembros, que es lo razonable aca: evita que
    # un unico par muy parecido arrastre a todo un grupo.
    enlace = linkage(squareform(distancia, checks=False), method="average")

    # Se corta el arbol a la altura que corresponde al umbral de redundancia:
    # con UMBRAL_FAMILIA = 0.80, se corta en distancia 0.20, de modo que
    # quedan juntas las metricas con |r| por encima de 0.80.
    etiquetas = fcluster(enlace, t=1 - UMBRAL_FAMILIA, criterion="distance")

    # Orden en que el agrupamiento deja las metricas. Sirve para dibujar la
    # matriz con las familias contiguas y que los bloques se vean.
    orden = leaves_list(enlace)

    return etiquetas, [matriz.columns[i] for i in orden]


def elegir_representantes(tabla_familias):
    """Elige una metrica por familia, segun el orden fijado en PRIORIDAD.

    Recorre PRIORIDAD de principio a fin y se queda con la primera metrica de
    cada familia que aparece. Asi la eleccion queda determinada por criterios
    escritos de antemano y no por los resultados de cada metrica.
    """
    elegidas = {}

    for metrica in PRIORIDAD:
        fila = tabla_familias[tabla_familias.metrica == metrica]
        if fila.empty:
            continue                      # metrica excluida del analisis
        familia = fila.familia.iloc[0]
        # Solo se guarda la primera de cada familia: las siguientes ya tienen
        # representante y son redundantes con ella.
        if familia not in elegidas:
            elegidas[familia] = metrica

    return elegidas


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

    # Solo las ventanas validas del analisis principal. Las de 10 s se
    # excluyen porque en ellas la mayoria de las metricas no se calcula.
    df = df[df.valida & (df.window_duration_s == DURACION)]

    # Se excluyen las metricas que quedaron vacias o constantes: sin
    # variacion no se puede calcular una correlacion.
    columnas = [m for m in METRICAS if df[m].notna().sum() > 0
                and df[m].std() > 0]

    # Matriz promediada entre participantes, y familias que surgen de ella.
    matriz, n_participantes = matriz_promedio(df, columnas)
    familias, orden = agrupar(matriz)

    print(f"Participantes            : {n_participantes}")
    print(f"Ventanas de {DURACION} s por persona : "
          f"{df.groupby('participant_id').size().median():.0f} (mediana)\n")

    tabla_familias = (pd.DataFrame({"metrica": matriz.columns,
                                    "familia": familias})
                      .sort_values(["familia", "metrica"]))

    # Representante de cada familia, segun el orden de PRIORIDAD.
    elegidas = elegir_representantes(tabla_familias)
    tabla_familias["representante"] = tabla_familias.apply(
        lambda f: elegidas.get(f.familia) == f.metrica, axis=1)

    print("=== familias y su representante ===")
    for numero, grupo in tabla_familias.groupby("familia"):
        otras = [m for m in grupo.metrica if m != elegidas[numero]]
        print(f"  Familia {numero}: {elegidas[numero]}"
              + (f"   (representa a: {', '.join(otras)})" if otras else "   (sola)"))

    print(f"\nDe {len(columnas)} metricas quedan "
          f"{tabla_familias.familia.nunique()} grupos independientes.")
    print(f"Metricas a reportar: {', '.join(elegidas.values())}")

    print("\n=== pares mas redundantes ===")
    # np.triu con k=1 se queda con el triangulo superior sin la diagonal, de
    # modo que cada par aparezca una sola vez y no se cuente la correlacion
    # de cada metrica consigo misma.
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
