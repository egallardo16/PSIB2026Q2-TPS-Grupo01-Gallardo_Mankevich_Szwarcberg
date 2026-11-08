"""procesar_registros.py -- CORRE TODO el pipeline sobre los 57 registros.

    QUE HACE : para cada participante, carga el ECG, lo filtra, detecta los
               QRS, arma la serie RR y la convierte en serie NN.
    ENTRA    : data/raw/VPxx/BitalinoECG.txt
    SALE     : data/processed/intervalos.csv          <- series RR y NN
               data/processed/eventos.csv             <- eventos del protocolo
               data/processed/calidad_deteccion.csv   <- indicadores de calidad
               data/processed/picos/VPxx.npy          <- para no recalcular
    CORRER   : python scripts/procesar_registros.py

--------------------------------------------------------------------------

Corre el pipeline de procesamiento sobre todos los registros y guarda los picos R.

Para cada participante: carga el ECG, lo filtra, detecta los complejos QRS con
Pan-Tompkins, arma la serie de intervalos RR y busca latidos ectopicos.

Detectar los QRS sobre los 57 registros lleva varios minutos, asi que los picos
R detectados se guardan en data/processed/picos/. Los pasos siguientes del
analisis los leen de ahi en lugar de volver a procesar la señal.

Imprime ademas un resumen por participante con indicadores de calidad, para
poder revisar de un vistazo si alguna deteccion salio mal.

Uso:
    python scripts/procesar_registros.py
    python scripts/procesar_registros.py --sujetos VP03 VP06
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Se agrega la raiz del repositorio al path para poder importar modulos/
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from modulos.eventos import cargar_eventos
from modulos.intervalos import construir_intervalos, resumen_calidad
from modulos.preprocesamiento import filtroPB
from modulos.procesamiento import (calcular_serie_rr, clasificar_ectopicos,
                                   detectar_qrs, panTompkins)

FS = 100                                  # frecuencia de muestreo del BITalino [Hz]
CRUDO = RAIZ / "data" / "raw"
PROCESADO = RAIZ / "data" / "processed"
PICOS = PROCESADO / "picos"


def cargar_ecg(codigo: str) -> np.ndarray:
    """Carga la señal de ECG de un participante desde data/raw/.

    Es equivalente a cargar_ECG_offline de modulos/preprocesamiento.py, pero
    lee de data/raw/VPxx/BitalinoECG.txt (el dataset completo descargado) en
    lugar de registros/VPxx.txt (los cinco registros de muestra).
    """
    df = pd.read_csv(CRUDO / codigo / "BitalinoECG.txt", sep="\t", header=None,
                     names=["voltaje", "tiempo", "raw"], usecols=["voltaje"])

    # errors="coerce" convierte en nulo cualquier valor no numerico, y dropna
    # los elimina. Hace falta porque algun archivo puede traer basura suelta
    # (por ejemplo registros/VP03.txt tiene dos caracteres de mas al inicio).
    voltaje = pd.to_numeric(df["voltaje"], errors="coerce").dropna()
    return voltaje.to_numpy()


def procesar(codigo: str) -> tuple[dict, pd.DataFrame]:
    """Procesa un registro.

    Devuelve sus indicadores de deteccion y su tabla de intervalos RR/NN.
    """
    ecg = cargar_ecg(codigo)
    duracion_min = len(ecg) / FS / 60

    # Filtro pasabanda 0.5-40 Hz: saca la deriva de linea de base y el ruido
    # de alta frecuencia, conservando la morfologia del complejo QRS.
    ecg_limpio = filtroPB(FS, ecg)

    # Pan-Tompkins devuelve la señal integrada (sobre la que se buscan los
    # picos) y la filtrada en 5-15 Hz (sobre la que se refina la posicion de R).
    ecg_integrado, ecg_filtrado = panTompkins(FS, ecg_limpio)
    picos_R = detectar_qrs(FS, ecg_integrado, ecg_filtrado)

    rr = calcular_serie_rr(picos_R, FS)
    ectopicos, _, latidos = clasificar_ectopicos(picos_R, FS)

    # Se guardan los picos R para no tener que repetir la deteccion.
    PICOS.mkdir(parents=True, exist_ok=True)
    np.save(PICOS / f"{codigo}.npy", picos_R)

    # Serie NN: descarta los intervalos que no superan los criterios de validez
    # definidos en modulos/intervalos.py.
    intervalos = construir_intervalos(picos_R, FS, codigo)

    # Se combinan los indicadores de la deteccion con los de la serie NN.
    indicadores = {
        "participante": codigo,
        "duracion_min": round(duracion_min, 1),
        "latidos": len(picos_R),
        "rr_mediana_s": round(float(np.median(rr)), 3),
        # El detector de ectopicos no encontro ninguno en este dataset; se deja
        # la columna para dejar constancia de que el control se hizo.
        "ectopicos": len(ectopicos),
        "latidos_analizados": len(latidos),
    }
    indicadores.update(resumen_calidad(intervalos))
    return indicadores, intervalos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sujetos", nargs="+", metavar="VPxx",
                    help="participantes a procesar; por defecto, todos")
    args = ap.parse_args()

    sujetos = args.sujetos or sorted(d.name for d in CRUDO.glob("VP*"))
    if not sujetos:
        print(f"No hay registros en {CRUDO}. Correr antes descargar_dataset.py")
        return 1

    filas, tablas, eventos = [], [], []
    for i, codigo in enumerate(sujetos, 1):
        inicio = time.time()
        indicadores, intervalos = procesar(codigo)
        filas.append(indicadores)
        tablas.append(intervalos)

        # Los eventos del protocolo se leen aparte, del Triggers.txt, y se
        # guardan en una tabla propia porque describen el experimento y no
        # la señal: son los mismos para cualquier analisis que se haga.
        ev = cargar_eventos(codigo)
        ev.insert(0, "participant_id", codigo)
        eventos.append(ev)

        print(f"[{i}/{len(sujetos)}] {codigo} ({time.time() - inicio:.1f} s)",
              flush=True)

    df = pd.DataFrame(filas)
    pd.set_option("display.width", 250)

    columnas = ["participante", "duracion_min", "latidos", "intervalos",
                "validos", "descartados", "pct_descartado", "fuera_de_rango",
                "salto_brusco", "nn_mediana_ms", "hr_media_lpm", "ectopicos"]

    print("\n" + "=" * 120)
    print(df[columnas].to_string(index=False))

    print("\n--- resumen ---")
    print(f"participantes procesados    : {len(df)}")
    print(f"latidos detectados          : {df.latidos.sum()}")
    print(f"intervalos totales          : {df.intervalos.sum()}")
    print(f"intervalos descartados      : {df.descartados.sum()} "
          f"({100 * df.descartados.sum() / df.intervalos.sum():.2f} %)")
    print(f"  por rango fisiologico     : {df.fuera_de_rango.sum()}")
    print(f"  por salto brusco          : {df.salto_brusco.sum()}")
    print(f"ectopicos totales           : {df.ectopicos.sum()}")
    print(f"HR media [lpm]              : {df.hr_media_lpm.mean():.1f} "
          f"(min {df.hr_media_lpm.min()}, max {df.hr_media_lpm.max()})")

    # Participantes con mucha señal descartada: conviene revisarlos antes de
    # incluirlos en el analisis.
    problematicos = df[df.pct_descartado > 5].sort_values("pct_descartado",
                                                          ascending=False)
    print(f"\nparticipantes con mas del 5 % descartado: {len(problematicos)}")
    if len(problematicos):
        print(problematicos[["participante", "intervalos", "descartados",
                             "pct_descartado"]].to_string(index=False))

    # Se guardan las dos salidas: la tabla de indicadores de calidad, que
    # resume el procesamiento, e intervalos.csv con las series RR y NN, que
    # es la entrada de la extraccion de caracteristicas.
    PROCESADO.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESADO / "calidad_deteccion.csv", index=False)
    pd.concat(tablas, ignore_index=True).to_csv(PROCESADO / "intervalos.csv",
                                                index=False)
    pd.concat(eventos, ignore_index=True).to_csv(PROCESADO / "eventos.csv",
                                                 index=False)

    print(f"\nTabla de calidad : data/processed/calidad_deteccion.csv")
    print(f"Intervalos       : data/processed/intervalos.csv")
    print(f"Eventos          : data/processed/eventos.csv")
    print(f"Picos R          : data/processed/picos/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
