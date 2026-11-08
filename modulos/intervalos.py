"""intervalos.py -- QUE LATIDOS valen y cuales se descartan.

    QUE HACE : convierte la serie RR en serie NN, descartando los intervalos
               que no pueden ser latidos reales (errores de deteccion).
    ENTRA    : los picos R detectados por Pan-Tompkins
    SALE     : tabla con rr_ms, nn_ms y quality_flag por cada intervalo
    CRITERIOS: rango 300-2000 ms (Task Force 1996) y variacion maxima del
               20 % respecto del intervalo anterior (criterio de Malik)

--------------------------------------------------------------------------

Construccion de la serie de intervalos NN a partir de los picos R detectados.

La serie RR es la diferencia entre picos R consecutivos. No todos esos
intervalos son latidos reales: un QRS detectado de mas produce un RR
imposiblemente corto, y un latido no detectado produce uno imposiblemente
largo. La serie NN (Normal-to-Normal) es la serie RR despues de descartar
esos intervalos.

Criterios de validez
--------------------
Se aplican los dos criterios habituales en la bibliografia de HRV:

1. Rango fisiologico absoluto: un intervalo valido esta entre 300 y 2000 ms,
   equivalente a una frecuencia cardiaca de entre 30 y 200 lpm.
   Referencia: Task Force of the European Society of Cardiology and the North
   American Society of Pacing and Electrophysiology (1996), "Heart rate
   variability: standards of measurement, physiological interpretation, and
   clinical use", Circulation 93(5):1043-1065.

2. Criterio relativo: un intervalo es sospechoso si difiere mas del 20 % del
   intervalo anterior. Un cambio asi entre dos latidos consecutivos no es
   fisiologicamente plausible en reposo y delata un error de deteccion.
   Es la regla conocida como criterio de Malik, usada entre otros por Kubios.

Los intervalos invalidos se DESCARTAN, no se interpolan. Interpolar inventa
valores intermedios que reducen artificialmente las metricas basadas en
diferencias sucesivas (rMSSD, pNN50), justamente las que mas interesan aca.
En su lugar se deja constancia de cuales se descartaron, y mas adelante se
exigira un minimo de intervalos validos por ventana.
"""

import numpy as np
import pandas as pd

# Limites del rango fisiologico, en milisegundos (Task Force, 1996).
NN_MINIMO_MS = 300.0     # 200 lpm
NN_MAXIMO_MS = 2000.0    # 30 lpm

# Maxima variacion relativa aceptada respecto del intervalo anterior (Malik).
VARIACION_MAXIMA = 0.20  # 20 %

# Etiquetas de la columna quality_flag.
FLAG_OK = "ok"
FLAG_FUERA_DE_RANGO = "fuera_de_rango"
FLAG_SALTO_BRUSCO = "salto_brusco"


def construir_intervalos(picos_R, fs: float, participante: str) -> pd.DataFrame:
    """Arma la tabla de intervalos de un participante a partir de sus picos R.

    Parametros
    ----------
    picos_R : indices de muestra de las ondas R (salida de detectar_qrs)
    fs      : frecuencia de muestreo [Hz]
    participante : codigo del participante, por ejemplo "VP03"

    Devuelve un DataFrame con las columnas acordadas para intervalos.csv:
        participant_id  codigo del participante
        rr_start_s      instante del pico R que abre el intervalo [s]
        rr_end_s        instante del pico R que lo cierra [s]
        rr_ms           duracion del intervalo [ms]
        nn_ms           igual a rr_ms si el intervalo es valido; NaN si no
        quality_flag    ok / fuera_de_rango / salto_brusco
    """
    picos_R = np.sort(np.asarray(picos_R, dtype=int))

    # Los tiempos se expresan en segundos desde el inicio de la grabacion,
    # igual que los eventos de modulos/eventos.py, para poder cruzarlos.
    tiempos_s = picos_R / fs

    # Cada intervalo va del pico i al pico i+1. De N picos salen N-1 intervalos.
    rr_ms = np.diff(tiempos_s) * 1000.0

    # --- Criterio 1: rango fisiologico absoluto ---
    fuera_de_rango = (rr_ms < NN_MINIMO_MS) | (rr_ms > NN_MAXIMO_MS)

    # --- Criterio 2: variacion relativa respecto del intervalo anterior ---
    # Se compara cada intervalo con el que lo precede. El primero no tiene
    # anterior con que compararse, asi que queda exento de este criterio.
    salto = np.zeros(len(rr_ms), dtype=bool)
    if len(rr_ms) > 1:
        variacion = np.abs(rr_ms[1:] - rr_ms[:-1]) / rr_ms[:-1]
        salto[1:] = variacion > VARIACION_MAXIMA

    # El criterio de rango tiene prioridad sobre el de salto: si un intervalo
    # es imposible en terminos absolutos, esa es la razon principal del descarte.
    quality_flag = np.full(len(rr_ms), FLAG_OK, dtype=object)
    quality_flag[salto] = FLAG_SALTO_BRUSCO
    quality_flag[fuera_de_rango] = FLAG_FUERA_DE_RANGO

    # La serie NN conserva el valor solo donde el intervalo es valido.
    # Los descartados quedan en NaN en lugar de desaparecer, para que la tabla
    # mantenga la correspondencia con los tiempos y se pueda auditar.
    nn_ms = np.where(quality_flag == FLAG_OK, rr_ms, np.nan)

    return pd.DataFrame({
        "participant_id": participante,
        "rr_start_s": np.round(tiempos_s[:-1], 3),
        "rr_end_s": np.round(tiempos_s[1:], 3),
        "rr_ms": np.round(rr_ms, 1),
        "nn_ms": np.round(nn_ms, 1),
        "quality_flag": quality_flag,
    })


def resumen_calidad(df: pd.DataFrame) -> dict:
    """Indicadores de calidad de la serie de un participante.

    Se usa para decidir si un registro es utilizable y para documentar en el
    informe cuanta señal se descarto.
    """
    total = len(df)
    validos = int((df.quality_flag == FLAG_OK).sum())

    return {
        "participant_id": df.participant_id.iloc[0],
        "intervalos": total,
        "validos": validos,
        "descartados": total - validos,
        "pct_descartado": round(100 * (total - validos) / total, 2) if total else 0.0,
        "fuera_de_rango": int((df.quality_flag == FLAG_FUERA_DE_RANGO).sum()),
        "salto_brusco": int((df.quality_flag == FLAG_SALTO_BRUSCO).sum()),
        "nn_mediana_ms": round(float(df.nn_ms.median()), 1),
        "hr_media_lpm": round(60_000 / df.nn_ms.mean(), 1) if validos else np.nan,
    }


if __name__ == "__main__":
    # Corre solo si se ejecuta este archivo directamente. Muestra los primeros
    # intervalos de un participante y su resumen de calidad, para ver de que
    # se trata sin tener que procesar los 57 registros.
    #
    #     python modulos/intervalos.py          -> VP03
    #     python modulos/intervalos.py VP70     -> el registro con mas descartes
    import sys
    from pathlib import Path

    codigo = sys.argv[1] if len(sys.argv) > 1 else "VP03"
    archivo = Path(__file__).resolve().parent.parent / "data" / "processed" / "picos" / f"{codigo}.npy"

    if not archivo.exists():
        print(f"No hay picos guardados de {codigo}.")
        print("Correr antes: python scripts/procesar_registros.py")
        sys.exit(1)

    tabla = construir_intervalos(np.load(archivo), fs=100, participante=codigo)

    print(f"Primeros 15 intervalos de {codigo}:\n")
    print(tabla.head(15).to_string(index=False))

    print(f"\nIntervalos descartados ({(tabla.quality_flag != FLAG_OK).sum()} en total):\n")
    descartados = tabla[tabla.quality_flag != FLAG_OK]
    print(descartados.head(10).to_string(index=False) if len(descartados)
          else "  ninguno: todos los intervalos pasaron los criterios")

    print("\nResumen de calidad:")
    for clave, valor in resumen_calidad(tabla).items():
        print(f"  {clave:<16} {valor}")
