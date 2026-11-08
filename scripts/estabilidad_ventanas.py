"""estabilidad_ventanas.py -- QUE METRICAS aguantan ventanas de 10 segundos.

    QUE HACE : mide cuanto cambia cada metrica entre los seis tramos de 10 s
               de un mismo minuto, para decidir cuales son interpretables en
               ventanas cortas y cuales solo en ventanas de 60 s.
    ENTRA    : data/processed/intervalos.csv y eventos.csv
    SALE     : results/tables/estabilidad_ventanas.csv
               results/figures/06_estabilidad_ventanas.png
    CORRER   : python scripts/estabilidad_ventanas.py

--------------------------------------------------------------------------

El razonamiento
---------------
Dentro de un mismo video de 60 s la condicion experimental no cambia. Si una
metrica fuera estable en ventanas de 10 s, los seis tramos de ese minuto
deberian dar valores parecidos. Si dan valores dispares, lo que estamos
midiendo a 10 s es mayormente ruido de estimacion.

Para que esa dispersion sea interpretable hay que compararla con algo. Se la
compara con la dispersion de la misma metrica ENTRE ventanas distintas, que
es la variacion que el analisis busca detectar:

    cociente = dispersion dentro de un minuto / dispersion entre minutos

Un cociente cercano a 1 o mayor significa que el ruido de estimar la metrica
con 10 s es tan grande como las diferencias que se quieren medir: esa metrica
no sirve a 10 s.

Se reporta ademas la correlacion entre el valor de un tramo de 10 s y el valor
del minuto que lo contiene. Responde a la pregunta practica: si tomo 10 s al
azar, cuanto me dice del minuto entero.

El analisis se hace sobre los participantes de Group3, que es el grupo sin
entrenamiento de biofeedback previo al reposo y el de mejor calidad de señal.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from modulos.metricas import METRICAS, calcular_todas

PROCESADO = RAIZ / "data" / "processed"
CRUDO = RAIZ / "data" / "raw"
FIGURAS = RAIZ / "results" / "figures"
TABLAS = RAIZ / "results" / "tables"

DURACION_VENTANA = 60      # duracion del video completo [s]
DURACION_SUBVENTANA = 10   # tramos en que se lo subdivide [s]
MINIMO_INTERVALOS = 5      # por debajo de esto no se calcula la sub-ventana


def cohorte_group3():
    """Participantes de Group3, el grupo sin biofeedback previo al reposo."""
    grupos = pd.read_csv(CRUDO / "subject_groups.csv")
    return sorted(grupos["Group3"].dropna().tolist())


def nn_en_rango(intervalos, inicio_s, fin_s):
    """Intervalos NN validos cuyo latido final cae dentro del rango dado."""
    dentro = intervalos[(intervalos.rr_end_s >= inicio_s) &
                        (intervalos.rr_end_s < fin_s)]
    return dentro.nn_ms.dropna().to_numpy()


def recolectar():
    """Calcula las metricas de cada video y de sus seis tramos de 10 s."""
    intervalos = pd.read_csv(PROCESADO / "intervalos.csv")
    eventos = pd.read_csv(PROCESADO / "eventos.csv")
    cohorte = cohorte_group3()

    filas_minuto, filas_tramo = [], []

    for participante in cohorte:
        iv = intervalos[intervalos.participant_id == participante]
        videos = eventos[(eventos.participant_id == participante) &
                         (eventos.condicion == "exposicion")]

        for _, video in videos.iterrows():
            # Se toman los primeros 60 s de cada video. Algunos duran 61 s, y
            # recortar todos al mismo largo evita que una ventana mas larga
            # parezca mas estable solo por tener mas latidos.
            inicio = video.inicio_s
            fin = inicio + DURACION_VENTANA

            nn_minuto = nn_en_rango(iv, inicio, fin)

            # Un minuto con menos de 30 intervalos quedo recortado o muy
            # sucio: no sirve ni como referencia ni para subdividir.
            if len(nn_minuto) < 30:
                continue

            # Valor de referencia: la metrica calculada con el minuto entero.
            metricas_minuto = calcular_todas(nn_minuto)
            metricas_minuto.update(participante=participante,
                                   evento=video.evento)
            filas_minuto.append(metricas_minuto)

            # El mismo minuto, ahora partido en seis tramos de 10 s. Como la
            # condicion experimental no cambia dentro del video, lo que
            # difiera entre estos seis tramos es error de estimacion.
            for k in range(DURACION_VENTANA // DURACION_SUBVENTANA):
                desde = inicio + k * DURACION_SUBVENTANA
                nn_tramo = nn_en_rango(iv, desde, desde + DURACION_SUBVENTANA)

                # Tramos con muy pocos latidos se saltean: sus metricas
                # serian ruido puro y exagerarian la dispersion medida.
                if len(nn_tramo) < MINIMO_INTERVALOS:
                    continue

                metricas_tramo = calcular_todas(nn_tramo)
                metricas_tramo.update(participante=participante,
                                      evento=video.evento, tramo=k)
                filas_tramo.append(metricas_tramo)

    return pd.DataFrame(filas_minuto), pd.DataFrame(filas_tramo)


def analizar(minutos, tramos):
    """Compara, metrica por metrica, la dispersion intra-minuto con la externa."""
    clave = ["participante", "evento"]
    filas = []

    for metrica in METRICAS:
        # Los conteos absolutos escalan con la duracion de la ventana: NN50
        # cuenta sobre ~13 diferencias en 10 s y sobre ~75 en 60 s. No son
        # magnitudes comparables entre ventanas, asi que el analisis no
        # aplica. Para eso estan sus versiones en proporcion, pNN50 y pNN20.
        if metrica == "NN50":
            filas.append({"metrica": metrica, "dispersion_intra": np.nan,
                          "dispersion_entre": np.nan, "cociente": np.nan,
                          "correlacion": np.nan})
            continue

        # Dispersion DENTRO de cada minuto: cuanto difieren entre si los seis
        # tramos de 10 s de un mismo video.
        dispersion_intra = tramos.groupby(clave)[metrica].std().median()

        # Dispersion ENTRE minutos DEL MISMO PARTICIPANTE. Se calcula por
        # persona y despues se toma la mediana, para que las diferencias de
        # pulso basal entre participantes no inflen la referencia: el analisis
        # compara condiciones dentro de cada persona, no personas entre si.
        dispersion_entre = minutos.groupby("participante")[metrica].std().median()

        # Correlacion entre el valor de cada tramo de 10 s y el del minuto que
        # lo contiene, tambien dentro de cada participante y promediada
        # despues, por la misma razon.
        pareado = tramos.merge(minutos, on=clave, suffixes=("_tramo", "_minuto"))
        correlacion = (pareado.groupby("participante")
                       .apply(lambda g: g[f"{metrica}_tramo"].corr(
                           g[f"{metrica}_minuto"]), include_groups=False)
                       .median())

        filas.append({
            "metrica": metrica,
            "dispersion_intra": round(dispersion_intra, 3),
            "dispersion_entre": round(dispersion_entre, 3),
            "cociente": round(dispersion_intra / dispersion_entre, 2)
                        if dispersion_entre else np.nan,
            "correlacion": round(correlacion, 3),
        })

    df = pd.DataFrame(filas).sort_values("cociente")
    # Criterio de lectura: un cociente por debajo de 1 significa que el ruido
    # de estimar con 10 s es menor que la señal que se quiere medir. Los
    # conteos absolutos, que no son comparables entre ventanas, quedan en NaN.
    df["apta_10s"] = np.where(df.cociente.isna(), "no aplica",
                              np.where(df.cociente < 1.0, "si", "no"))
    return df


def graficar(df):
    """Barras del cociente por metrica, con la linea de decision en 1."""
    df = df.dropna(subset=["cociente"]).sort_values("cociente")
    colores = ["tab:green" if a == "si" else "tab:red" for a in df.apta_10s]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df.metrica, df.cociente, color=colores)
    ax.axvline(1.0, color="k", linestyle="--", linewidth=1,
               label="ruido = señal")

    ax.set_xlabel("Dispersión dentro de un minuto / dispersión entre minutos")
    ax.set_title("Estabilidad de cada métrica en ventanas de 10 s\n"
                 "(verde: el ruido de estimación es menor que la señal a medir)")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def main() -> int:
    """Compara cada metrica entre tramos de 10 s y su minuto, y reporta el cociente."""
    if not (PROCESADO / "intervalos.csv").exists():
        print("Falta intervalos.csv. Correr antes: python scripts/procesar_registros.py")
        return 1

    minutos, tramos = recolectar()
    print(f"Ventanas de {DURACION_VENTANA} s analizadas : {len(minutos)}")
    print(f"Tramos de {DURACION_SUBVENTANA} s analizados    : {len(tramos)}")
    print(f"Intervalos NN por tramo de 10 s   : "
          f"mediana {tramos.n_intervalos.median():.0f}, "
          f"rango {tramos.n_intervalos.min()}-{tramos.n_intervalos.max()}\n")

    df = analizar(minutos, tramos)
    print(df.to_string(index=False))

    TABLAS.mkdir(parents=True, exist_ok=True)
    FIGURAS.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLAS / "estabilidad_ventanas.csv", index=False)
    graficar(df).savefig(FIGURAS / "06_estabilidad_ventanas.png", dpi=150,
                         bbox_inches="tight")

    aptas = df[df.apta_10s == "si"].metrica.tolist()
    print(f"\nAptas para ventanas de 10 s ({len(aptas)}): {', '.join(aptas)}")
    print(f"Solo para 60 s ({len(df) - len(aptas)}): "
          f"{', '.join(df[df.apta_10s == 'no'].metrica)}")
    print("\nTabla   : results/tables/estabilidad_ventanas.csv")
    print("Figura  : results/figures/06_estabilidad_ventanas.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
