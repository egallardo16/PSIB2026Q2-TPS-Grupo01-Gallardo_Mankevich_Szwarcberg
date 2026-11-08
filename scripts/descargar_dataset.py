"""descargar_dataset.py -- TRAE LOS DATOS de PhysioNet a tu computadora.

    QUE HACE : descarga el ECG y las marcas de eventos de los participantes.
    ENTRA    : nada (se conecta a PhysioNet)
    SALE     : data/raw/VPxx/BitalinoECG.txt y data/raw/VPxx/Triggers.txt
    CORRER   : python scripts/descargar_dataset.py       (~375 MB, varios min)
               python scripts/descargar_dataset.py --solo-triggers   (24 KB)

--------------------------------------------------------------------------

Descarga el dataset de ECG desde PhysioNet a la carpeta data/raw/.

El dataset pesa alrededor de 375 MB, por lo que no se versiona en el
repositorio: se baja con este script. Los cinco registros que estan en
registros/ alcanzan para abrir el notebook exploratorio sin descargar nada.

Dataset: "Electrocardiogram, skin conductance and respiration from
spider-fearful individuals watching spider video clips", version 1.0.0.
https://physionet.org/content/ecg-spider-clip/1.0.0/

Por cada participante baja dos archivos:
  - BitalinoECG.txt : la senal de ECG (voltaje, timestamp, etiqueta)
  - Triggers.txt    : inicio y fin de cada evento del experimento

Ejemplos de uso
---------------
    # todo el dataset (57 participantes, ~375 MB)
    python scripts/descargar_dataset.py

    # solo las marcas temporales (24 KB, util para explorar el protocolo)
    python scripts/descargar_dataset.py --solo-triggers

    # solo algunos participantes
    python scripts/descargar_dataset.py --sujetos VP03 VP06 VP09

Si la descarga falla, se puede bajar el ZIP a mano desde la pagina del
dataset y descomprimirlo dentro de data/raw/, respetando la estructura
data/raw/VP03/BitalinoECG.txt.
"""

import argparse
import sys
from pathlib import Path

import requests

BASE = "https://physionet.org/files/ecg-spider-clip/1.0.0"
DESTINO = Path(__file__).resolve().parent.parent / "data" / "raw"

# Los 57 participantes con registros disponibles. El estudio original tuvo 80,
# pero PhysioNet publica solo estos.
PARTICIPANTES = [
    "VP02", "VP03", "VP05", "VP06", "VP08", "VP09", "VP11", "VP12", "VP14",
    "VP15", "VP17", "VP18", "VP20", "VP23", "VP24", "VP26", "VP27", "VP29",
    "VP30", "VP32", "VP33", "VP35", "VP36", "VP38", "VP39", "VP41", "VP42",
    "VP44", "VP45", "VP47", "VP48", "VP50", "VP51", "VP53", "VP54", "VP56",
    "VP57", "VP59", "VP61", "VP62", "VP63", "VP64", "VP65", "VP66", "VP68",
    "VP69", "VP70", "VP71", "VP72", "VP73", "VP74", "VP75", "VP76", "VP77",
    "VP78", "VP79", "VP80",
]


def descargar(url: str, destino: Path) -> str:
    """Baja url a destino y devuelve un texto con el resultado.

    Si el archivo ya existe y no esta vacio no lo vuelve a bajar, de modo que
    el script se puede correr varias veces sin desperdiciar tiempo ni datos.
    """
    if destino.exists() and destino.stat().st_size > 0:
        return "ya estaba"

    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        # requests sigue las redirecciones de PhysioNet (los archivos se
        # sirven desde S3) y trae su propio paquete de certificados, lo que
        # evita el error CERTIFICATE_VERIFY_FAILED tipico de Python en macOS.
        respuesta = requests.get(url, timeout=120)
        respuesta.raise_for_status()
    except requests.RequestException as e:
        return f"ERROR: {e}"

    # Se escribe a un archivo temporal y despues se renombra, para que una
    # descarga interrumpida no deje un archivo a medias que el script crea
    # completo en la proxima corrida.
    temporal = destino.with_suffix(destino.suffix + ".parcial")
    temporal.write_bytes(respuesta.content)
    temporal.replace(destino)
    return f"ok ({tamano_legible(len(respuesta.content))})"


def tamano_legible(bytes_: int) -> str:
    """Formatea un tamano en la unidad que corresponda."""
    if bytes_ < 1024:
        return f"{bytes_} B"
    if bytes_ < 1024 ** 2:
        return f"{bytes_ / 1024:.1f} KB"
    return f"{bytes_ / 1024 ** 2:.1f} MB"


def main() -> int:
    """Descarga los archivos pedidos, salteando los que ya estan en disco."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--solo-triggers", action="store_true",
                    help="baja unicamente los Triggers.txt (24 KB en total)")
    ap.add_argument("--sujetos", nargs="+", metavar="VPxx",
                    help="lista de participantes; por defecto, los 57")
    args = ap.parse_args()

    sujetos = args.sujetos or PARTICIPANTES
    desconocidos = [s for s in sujetos if s not in PARTICIPANTES]
    if desconocidos:
        print(f"Participantes inexistentes en el dataset: {desconocidos}")
        return 1

    archivos = ["Triggers.txt"]
    if not args.solo_triggers:
        archivos.append("BitalinoECG.txt")

    tareas = [(s, a) for s in sujetos for a in archivos]
    print(f"Descargando {len(tareas)} archivos de {len(sujetos)} participantes "
          f"a {DESTINO}\n")

    fallidos = []
    for i, (sujeto, archivo) in enumerate(tareas, 1):
        estado = descargar(f"{BASE}/{sujeto}/{archivo}", DESTINO / sujeto / archivo)
        print(f"[{i}/{len(tareas)}] {sujeto}/{archivo}: {estado}", flush=True)
        if estado.startswith("ERROR"):
            fallidos.append(f"{sujeto}/{archivo}")

    # Tabla con la asignacion de participantes a los tres grupos del estudio.
    estado = descargar(f"{BASE}/subject_groups.csv", DESTINO / "subject_groups.csv")
    print(f"subject_groups.csv: {estado}")

    if fallidos:
        print(f"\nFallaron {len(fallidos)} archivos: {fallidos}")
        print("Volver a correr el script reintenta solo los que faltan.")
        return 1

    print("\nDescarga completa.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
