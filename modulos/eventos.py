"""eventos.py -- CUANDO paso cada cosa en el experimento.

    QUE HACE : traduce las marcas temporales del experimento a segundos desde
               el inicio de la grabacion, y etiqueta cada evento.
    ENTRA    : data/raw/VPxx/Triggers.txt (y la 1ra linea del ECG, para el t0)
    SALE     : tabla con evento, condicion, inicio_s, fin_s, duracion_s, orden
    SE USA   : para recortar la señal por video y para el analisis de habituacion

--------------------------------------------------------------------------

Carga de las marcas temporales de los eventos del experimento (Triggers.txt).

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

import re                      # expresiones regulares, para reconocer "CLIP-<numero>"
from pathlib import Path       # manejo de rutas independiente del sistema operativo

import pandas as pd            # tablas (DataFrames)

# Las rutas se arman a partir de la ubicacion de este mismo archivo, de modo que
# funcionen sin importar desde que carpeta se ejecute el codigo.
# __file__ es modulos/eventos.py -> .parent es modulos/ -> .parent.parent es la raiz.
RAIZ = Path(__file__).resolve().parent.parent
CRUDO = RAIZ / "data" / "raw"

# Nombres de los dos eventos que el analisis trata de forma especial. Se definen
# como constantes para no repetirlos como texto suelto por todo el archivo.
EVENTO_REPOSO = "BIOFEEDBACK-REST"   # descanso final, se usa como linea de base
EVENTO_DEMO = "CLIP-DEMO"            # video de demostracion, no es un estimulo fobico


def _a_segundos(hhmmss) -> float:
    """Convierte una hora hhmmss (o hhmmss.mmm) a segundos desde la medianoche.

    Por ejemplo, 144725 (14:47:25) devuelve 53245.0 segundos.

    El guion bajo inicial indica que es una funcion de uso interno de este
    modulo: no esta pensada para llamarse desde afuera.
    """
    # Se separan parte entera y decimales. Los triggers son enteros, pero los
    # timestamps del ECG traen milisegundos (144118.723) y hay que conservarlos.
    entero = int(float(hhmmss))
    fraccion = float(hhmmss) - entero

    # zfill(6) completa con ceros a la izquierda hasta tener 6 digitos. Hace
    # falta porque una hora como 09:47:25 llega como el numero 94725, y sin el
    # relleno se leeria mal al cortarlo por posicion.
    s = str(entero).zfill(6)

    # Se corta el texto en tres pares: [:2] horas, [2:4] minutos, [4:6] segundos.
    return int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) + fraccion


def _clasificar(evento: str) -> str:
    """Agrupa cada evento en la condicion experimental que representa."""
    # fullmatch exige que el nombre entero sea "CLIP-" seguido de digitos.
    # Asi CLIP-1 ... CLIP-16 entran como exposicion, pero CLIP-DEMO no, porque
    # DEMO no son digitos. Por eso se usa una expresion regular y no startswith.
    if re.fullmatch(r"CLIP-\d+", evento):
        return "exposicion"      # uno de los 16 videos de arañas
    if evento == EVENTO_DEMO:
        return "demo"            # video de demostracion inicial
    if evento == EVENTO_REPOSO:
        return "reposo"          # descanso final de 300 s, sirve de basal
    # Todo lo que no cayo en los casos anteriores es un entrenamiento de
    # biofeedback, que solo aparece en los participantes de Group2 y Group4.
    return "biofeedback"


def inicio_registro(codigo: str) -> float:
    """Instante en que arranca el ECG, en segundos desde la medianoche.

    Es el punto cero contra el que se miden todos los eventos.
    """
    # Se lee una unica linea en lugar del archivo entero: los ECG pesan unos
    # 6 MB y aca solo interesa el primer timestamp.
    with open(CRUDO / codigo / "BitalinoECG.txt") as f:
        primera_linea = f.readline()

    # Las columnas del ECG son: voltaje, timestamp, etiqueta. Queremos la 2da.
    return _a_segundos(primera_linea.split("\t")[1])


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
    # header=None porque el archivo no tiene fila de titulos; los nombres de
    # columna se los ponemos nosotras.
    df = pd.read_csv(CRUDO / codigo / "Triggers.txt", sep="\t", header=None,
                     names=["evento", "inicio", "fin"])

    # Punto cero: el instante en que arranco la grabacion de este participante.
    t0 = inicio_registro(codigo)

    # .map aplica la funcion a cada fila de la columna. Al restar t0, las horas
    # de reloj se convierten en segundos desde el inicio del registro, que es lo
    # que hace falta para recortar la señal.
    df["inicio_s"] = df["inicio"].map(_a_segundos) - t0
    df["fin_s"] = df["fin"].map(_a_segundos) - t0
    df["duracion_s"] = df["fin_s"] - df["inicio_s"]
    df["condicion"] = df["evento"].map(_clasificar)

    # Se ordena cronologicamente. Es imprescindible para el paso siguiente:
    # el orden de exposicion se deduce de la posicion en la tabla.
    df = df.sort_values("inicio_s").reset_index(drop=True)

    # es_video es una columna de True/False, una por fila, que marca cuales son
    # videos de arañas.
    es_video = df["condicion"] == "exposicion"

    # Se numeran del 1 al 16 solo esas filas. Las demas quedan en NaN (vacias)
    # porque la Serie que se asigna solo tiene indices para las filas de video.
    # Como la tabla ya esta ordenada por tiempo, ese numero es el orden real en
    # que el participante vio cada video, que es lo que se necesita para
    # analizar la habituacion.
    df["orden"] = pd.Series(range(1, es_video.sum() + 1), index=df.index[es_video])

    # Se devuelven solo las columnas utiles: las originales "inicio" y "fin",
    # que estaban en hora de reloj, ya no hacen falta.
    return df[["evento", "condicion", "inicio_s", "fin_s", "duracion_s", "orden"]]


if __name__ == "__main__":
    # Este bloque corre unicamente si se ejecuta este archivo de forma directa
    # (con el boton de play o con "python modulos/eventos.py"). Si otro archivo
    # importa este modulo, no se ejecuta.
    import sys

    # Permite elegir el participante por argumento: python modulos/eventos.py VP06
    codigo = sys.argv[1] if len(sys.argv) > 1 else "VP03"
    print(f"Eventos de {codigo}:\n")
    print(cargar_eventos(codigo).to_string(index=False))
