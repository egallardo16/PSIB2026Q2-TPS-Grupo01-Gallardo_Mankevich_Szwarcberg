"""ventanas.py -- CORTA la señal por evento y calcula las metricas de cada tramo.

    QUE HACE : divide la serie NN de cada participante en ventanas alineadas
               con los eventos del protocolo y calcula las metricas de cada una.
    ENTRA    : data/processed/intervalos.csv y eventos.csv
    SALE     : data/features/metricas.csv (una fila por ventana)
    CORRER   : python modulos/ventanas.py

--------------------------------------------------------------------------

Dos duraciones de ventana
-------------------------
- 60 s: una ventana por video. Es el analisis principal.
- 10 s: seis ventanas por video. Analisis secundario, limitado a las metricas
  que resultaron estables en scripts/estabilidad_ventanas.py.

Ese analisis midio, para cada metrica, cuanto varia entre los seis tramos de
10 s de un mismo video (donde la condicion no cambia, asi que esa variacion es
error de estimacion) frente a cuanto varia entre videos distintos del mismo
participante (que es lo que el analisis quiere detectar). Solo cuatro metricas
quedaron con el error por debajo de la señal. El resto se calcula unicamente
en las ventanas de 60 s.

Normalizacion por el reposo
---------------------------
Nmean (ecuacion 1 del articulo) se expresa respecto de la frecuencia cardiaca
media del propio participante durante el reposo. Esto permite comparar
personas con pulso basal distinto: lo que importa no es si alguien esta a 90
lpm, sino cuanto se aparta de su propio nivel de reposo.

Advertencia sobre el basal: el reposo del protocolo es el evento
BIOFEEDBACK-REST, que esta al final del experimento y no antes, por lo que no
es una linea de base previa a la exposicion. Ademas, en Group2 y Group4 ese
reposo viene precedido por un entrenamiento de biofeedback respiratorio, que
modifica la variabilidad cardiaca por diseño. Por eso el analisis principal se
restringe a Group3, el unico grupo sin ese entrenamiento.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from modulos.metricas import METRICAS, calcular_todas, frecuencia_instantanea

PROCESADO = RAIZ / "data" / "processed"
CARACTERISTICAS = RAIZ / "data" / "features"
CRUDO = RAIZ / "data" / "raw"

DURACIONES = (60, 10)        # duraciones de ventana a generar [s]

# Metricas que el analisis de estabilidad encontro utilizables en 10 s.
# Para las ventanas de 10 s, el resto se deja en NaN en lugar de calcularse,
# para que no se usen por error en el analisis posterior.
METRICAS_10S = {"std", "Nmean", "avNN", "rMSSD"}

# Una ventana se considera valida si conserva al menos esta fraccion de los
# intervalos que caberian en ella segun el pulso del propio participante.
# Con menos, la ventana quedo demasiado recortada por el control de calidad
# como para que sus metricas sean representativas.
FRACCION_MINIMA = 0.70


def cohorte_group3():
    """Participantes de Group3: sin biofeedback previo al reposo."""
    grupos = pd.read_csv(CRUDO / "subject_groups.csv")
    return sorted(grupos["Group3"].dropna().tolist())


def serie_en_rango(intervalos, inicio_s, fin_s):
    """Intervalos NN cuyo latido final cae en [inicio_s, fin_s).

    Devuelve la columna nn_ms CON los NaN de los intervalos descartados: las
    metricas los necesitan para no calcular diferencias entre latidos que no
    eran realmente consecutivos.
    """
    # Se usa rr_end_s (el instante del latido que CIERRA el intervalo) como
    # criterio de pertenencia, de modo que cada intervalo caiga en una sola
    # ventana y no se cuente dos veces en el borde.
    dentro = intervalos[(intervalos.rr_end_s >= inicio_s) &
                        (intervalos.rr_end_s < fin_s)]

    # Se devuelve nn_ms y no rr_ms: los intervalos que no pasaron el control
    # de calidad vienen como NaN en esa columna, y las metricas los necesitan
    # asi para no calcular diferencias entre latidos no consecutivos.
    return dentro.nn_ms.to_numpy()


def hr_basal(intervalos, eventos):
    """Frecuencia cardiaca media del participante durante el reposo [lpm].

    Devuelve NaN si el participante no tiene evento de reposo, como ocurre con
    VP26. Sin basal no se puede calcular Nmean.
    """
    reposo = eventos[eventos.condicion == "reposo"]
    if reposo.empty:
        return np.nan

    # Se toma el reposo completo, no una ventana: cuantos mas latidos entren,
    # mas estable queda la referencia contra la que se compara todo lo demas.
    nn = serie_en_rango(intervalos, reposo.inicio_s.iloc[0], reposo.fin_s.iloc[0])

    # El basal se expresa en lpm porque Nmean trabaja sobre la frecuencia
    # cardiaca instantanea, no sobre los intervalos.
    hr = frecuencia_instantanea(nn)

    # Con uno o ningun latido valido no hay promedio que valga la pena.
    if np.sum(~np.isnan(hr)) < 2:
        return np.nan

    # nanmean ignora los latidos descartados por el control de calidad.
    return float(np.nanmean(hr))


def ventanas_de_evento(evento, duracion_s):
    """Instantes de inicio de las ventanas que entran en un evento.

    Los videos duran 60 u 61 s segun el participante; se toman solo las
    ventanas completas, para que todas tengan la misma duracion y sean
    comparables entre si.
    """
    cuantas = int((evento.fin_s - evento.inicio_s) // duracion_s)
    return [evento.inicio_s + k * duracion_s for k in range(cuantas)]


def metricas_de_participante(intervalos, eventos, participante, basal):
    """Todas las ventanas de un participante, en las dos duraciones."""
    filas = []

    # Cuantos intervalos caben por segundo en esta persona. Se calcula con la
    # mediana de sus propios NN, porque el umbral de validez tiene que
    # adaptarse a su pulso: en alguien de 60 lpm entran la mitad de latidos
    # en diez segundos que en alguien de 120.
    nn_mediana = np.nanmedian(intervalos.nn_ms.to_numpy())
    por_segundo = 1000.0 / nn_mediana if nn_mediana > 0 else 0

    # Se recorren las dos duraciones de ventana: 60 s y 10 s.
    for duracion in DURACIONES:
        # Cantidad minima de intervalos para aceptar una ventana de este largo.
        minimo = FRACCION_MINIMA * por_segundo * duracion

        # Y dentro de cada duracion, todos los eventos del protocolo: los 16
        # videos, el de demostracion y el reposo.
        for _, evento in eventos.iterrows():
            for inicio in ventanas_de_evento(evento, duracion):
                # Los intervalos NN que caen en esta ventana, con los huecos.
                nn = serie_en_rango(intervalos, inicio, inicio + duracion)

                # El basal solo lo usa Nmean; las demas lo ignoran.
                metricas = calcular_todas(nn, hr_basal=basal)

                # En las ventanas cortas se anulan las metricas que el
                # analisis de estabilidad mostro dominadas por el ruido. Se
                # dejan en NaN en vez de no guardarlas, para que la tabla
                # tenga siempre las mismas columnas y quede explicito que no
                # es que falten: es que no son interpretables a 10 s.
                if duracion == 10:
                    for nombre in METRICAS:
                        if nombre not in METRICAS_10S:
                            metricas[nombre] = np.nan

                filas.append({
                    "participant_id": participante,
                    "event_id": evento.evento,
                    "condition": evento.condicion,
                    "exposure_order": evento.orden,
                    "window_start_s": round(inicio, 3),
                    "window_duration_s": duracion,
                    **metricas,
                    "valida": metricas["n_intervalos"] >= minimo,
                })

    return filas


def generar(participantes=None):
    """Arma la tabla de metricas de todos los participantes indicados."""
    # Las dos tablas que dejo el procesamiento de la señal.
    intervalos = pd.read_csv(PROCESADO / "intervalos.csv")
    eventos = pd.read_csv(PROCESADO / "eventos.csv")
    participantes = participantes or cohorte_group3()

    filas = []
    for participante in participantes:
        # Se recortan las dos tablas a este participante antes de ventanear.
        iv = intervalos[intervalos.participant_id == participante]
        ev = eventos[eventos.participant_id == participante]

        # El basal se calcula una sola vez por persona y vale para todas sus
        # ventanas: es su nivel de reposo, no algo propio de cada ventana.
        basal = hr_basal(iv, ev)

        filas.extend(metricas_de_participante(iv, ev, participante, basal))

    # Se fija el orden de las columnas para que el CSV salga siempre igual:
    # primero las que identifican la ventana, despues las metricas.
    columnas = (["participant_id", "event_id", "condition", "exposure_order",
                 "window_start_s", "window_duration_s"]
                + list(METRICAS) + ["n_intervalos", "valida"])
    return pd.DataFrame(filas)[columnas]


if __name__ == "__main__":
    if not (PROCESADO / "intervalos.csv").exists():
        print("Falta intervalos.csv. Correr antes: "
              "python scripts/procesar_registros.py")
        sys.exit(1)

    tabla = generar()

    CARACTERISTICAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CARACTERISTICAS / "metricas.csv", index=False)

    print(f"Participantes : {tabla.participant_id.nunique()}")
    print(f"Ventanas      : {len(tabla)}\n")

    resumen = (tabla.groupby(["window_duration_s", "condition"])
               .agg(ventanas=("valida", "size"),
                    validas=("valida", "sum"),
                    nn_mediana=("n_intervalos", "median"))
               .reset_index())
    print(resumen.to_string(index=False))

    print(f"\nVentanas no validas: {(~tabla.valida).sum()} de {len(tabla)} "
          f"({100 * (~tabla.valida).sum() / len(tabla):.1f} %)")
    print(f"\nGuardado en data/features/metricas.csv")
