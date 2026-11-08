"""Carga de las marcas temporales de los eventos del experimento (Triggers.txt).

Cada participante tiene un Triggers.txt con tres columnas separadas por
tabulaciones: el nombre del evento, y su hora de inicio y de fin en formato
hhmmss. Por ejemplo:

    CLIP-DEMO            144725  144835
    CLIP-4               144852  144953
    BIOFEEDBACK-REST     151146  151647

Esas horas son horas del reloj, no segundos desde el comienzo del registro.
Para poder recortar la señal hay que restarles el instante en que arranco la
grabacion, que esta en la segunda columna del ECG. De eso se encarga
`cargar_eventos`.
"""

import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
CRUDO = RAIZ / "data" / "raw"

EVENTO_REPOSO = "BIOFEEDBACK-REST"
EVENTO_DEMO = "CLIP-DEMO"


def _a_segundos(hhmmss) -> float:
    """Convierte una hora hhmmss (o hhmmss.mmm) a segundos desde la medianoche."""
    entero = int(float(hhmmss))
    fraccion = float(hhmmss) - entero
    s = str(entero).zfill(6)
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) + fraccion


def _clasificar(evento: str) -> str:
    """Agrupa cada evento en la condicion experimental que representa."""
    if re.fullmatch(r"CLIP-\d+", evento):
        return "exposicion"      # uno de los 16 videos de arañas
    if evento == EVENTO_DEMO:
        return "demo"            # video de demostracion inicial
    if evento == EVENTO_REPOSO:
        return "reposo"          # descanso final de 300 s, sirve de basal
    return "biofeedback"         # entrenamiento, solo en Group2 y Group4


def inicio_registro(codigo: str) -> float:
    """Instante en que arranca el ECG, en segundos desde la medianoche.

    Lee solo la primera linea del archivo: los ECG pesan ~6 MB y no hace falta
    cargarlos enteros para esto.
    """
    with open(CRUDO / codigo / "BitalinoECG.txt") as f:
        return _a_segundos(f.readline().split("\t")[1])


def cargar_eventos(codigo: str) -> pd.DataFrame:
    """Devuelve los eventos del participante, en segundos desde el inicio del ECG.

    Columnas:
        evento      nombre original del trigger (CLIP-4, BIOFEEDBACK-REST, ...)
        condicion   exposicion / demo / reposo / biofeedback
        inicio_s    segundos desde que arranco la grabacion
        fin_s       idem
        duracion_s  fin_s - inicio_s
        orden       orden de aparicion de los 16 videos (1 a 16); NaN en el resto

    El orden se calcula a partir de los tiempos y no del numero de video,
    porque los videos se presentaron en orden aleatorio para cada participante.
    """
    df = pd.read_csv(CRUDO / codigo / "Triggers.txt", sep="\t", header=None,
                     names=["evento", "inicio", "fin"])

    t0 = inicio_registro(codigo)
    df["inicio_s"] = df["inicio"].map(_a_segundos) - t0
    df["fin_s"] = df["fin"].map(_a_segundos) - t0
    df["duracion_s"] = df["fin_s"] - df["inicio_s"]
    df["condicion"] = df["evento"].map(_clasificar)

    df = df.sort_values("inicio_s").reset_index(drop=True)

    es_video = df["condicion"] == "exposicion"
    df["orden"] = pd.Series(range(1, es_video.sum() + 1), index=df.index[es_video])

    return df[["evento", "condicion", "inicio_s", "fin_s", "duracion_s", "orden"]]


if __name__ == "__main__":
    # Ejecutar este archivo directamente muestra los eventos de un participante,
    # para verificar a simple vista que la carga funciona.
    import sys

    codigo = sys.argv[1] if len(sys.argv) > 1 else "VP03"
    print(f"Eventos de {codigo}:\n")
    print(cargar_eventos(codigo).to_string(index=False))
