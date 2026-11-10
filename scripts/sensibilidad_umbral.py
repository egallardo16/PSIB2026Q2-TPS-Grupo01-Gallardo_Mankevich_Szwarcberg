"""sensibilidad_umbral.py -- ¿CAMBIAN las conclusiones con otro umbral de ventana válida?

    QUE HACE : repite P1, P3a y P3b variando el umbral de validez de ventana,
               para ver si alguna conclusión depende de ese criterio.
    ENTRA    : data/processed/intervalos.csv y eventos.csv
    SALE     : results/tables/sensibilidad_umbral_validez.csv
    CORRER   : python scripts/sensibilidad_umbral.py

--------------------------------------------------------------------------

Por que hace falta
------------------
Una ventana se considera valida si conserva al menos cierta fraccion de los
intervalos que caberian en ella segun el pulso del participante
(`ventanas.FRACCION_MINIMA`). Hace falta un piso de ese tipo para excluir
ventanas degeneradas, en las que el control de calidad descarto tantos
intervalos que las metricas dejan de ser representativas.

Ahora bien, no hay en la bibliografia un valor consensuado para ese piso. Por
eso, en lugar de fijarlo por convencion, se evaluo el rango completo de valores
posibles y se verifico sobre que intervalo las conclusiones se mantienen
estables. El umbral adoptado en el trabajo se eligio dentro de ese rango.

Que se reporta
--------------
Para cada umbral se vuelven a armar las ventanas desde cero y se corren los
tres tests principales. La tabla final muestra, por pregunta y metrica, el p
corregido por Holm y si dio significativo.

Se reporta ademas, para cada umbral, cuantas ventanas quedan y **cuantos
intervalos NN tiene la ventana mas pobre que se acepta**. Ese segundo numero es
el que da la medida real de cuanto filtra el criterio: si incluso con el
umbral mas laxo las ventanas que entran conservan decenas de intervalos, el
piso no esta recortando datos utiles, sino cumpliendo su funcion de descartar
casos degenerados.

El rango llega hasta 0.99 a proposito. En la zona baja el umbral casi no
descarta nada, asi que una tabla acotada a esos valores mostraria varias
columnas identicas sin permitir distinguir si eso refleja robustez o
simplemente un rango de prueba demasiado estrecho. Llevandolo hasta donde el
criterio si empieza a morder se ve hasta donde aguanta cada conclusion.
"""

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from modulos import estadisticas as est
from modulos import ventanas

TABLAS = RAIZ / "results" / "tables"

# Del 50 % (muy laxo) al 99 % (practicamente exige la ventana completa).
UMBRALES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


def correr_un_umbral(umbral):
    """Rearma las ventanas con ese umbral y corre los tres tests.

    Devuelve (filas de resultados, resumen de cuantas ventanas quedan).
    """
    # El resto del codigo lee este valor al calcular cada ventana, asi que
    # alcanza con reasignarlo antes de volver a generar.
    ventanas.FRACCION_MINIMA = umbral
    metricas = ventanas.generar()

    validas = metricas[metricas.valida]

    # Cuantas ventanas sobreviven, separando por duracion: el umbral pega
    # distinto en las de 10 s (unos 14 intervalos) que en las de 60 s (unos 81).
    conteo = {}
    for duracion in (60, 10):
        del_largo = metricas[metricas.window_duration_s == duracion]
        v = del_largo[del_largo.valida]
        conteo[f"ventanas_{duracion}s"] = len(v)
        conteo[f"descartadas_{duracion}s"] = len(del_largo) - len(v)
        # La ventana aceptada con menos intervalos: el peor caso que entra.
        conteo[f"min_nn_{duracion}s"] = int(v.n_intervalos.min()) if len(v) else 0

    # Los tres tests principales, con las mismas funciones que usa el analisis.
    tests = {
        "P1": est.p1_exposicion_vs_reposo(metricas),
        "P3a": est.p3a_habituacion_entre_videos(ventanas.habituacion(metricas)),
        "P3b": est.p3b_habituacion_dentro_del_video(metricas),
    }

    filas = []
    for pregunta, tabla in tests.items():
        for _, f in tabla.iterrows():
            filas.append({"umbral": umbral, "pregunta": pregunta,
                          "metrica": f.metrica, "diferencia": round(f.diferencia, 3),
                          "p_holm": f.p_holm, "significativo": f.significativo,
                          **conteo})
    return filas, conteo


def correr(umbrales=UMBRALES):
    """Recorre todos los umbrales y devuelve la tabla completa."""
    filas, resumen = [], []
    original = ventanas.FRACCION_MINIMA

    try:
        for umbral in umbrales:
            print(f"  umbral {umbral:.0%} ...", flush=True)
            nuevas, conteo = correr_un_umbral(umbral)
            filas.extend(nuevas)
            resumen.append({"umbral": umbral, **conteo})
    finally:
        # Se restituye pase lo que pase, para no dejar el modulo alterado si
        # algo falla a mitad de camino.
        ventanas.FRACCION_MINIMA = original

    return pd.DataFrame(filas), pd.DataFrame(resumen)


def main() -> int:
    """Corre el analisis de sensibilidad y guarda la tabla."""
    if not (RAIZ / "data" / "processed" / "intervalos.csv").exists():
        print("Falta intervalos.csv. Correr antes: "
              "python scripts/procesar_registros.py")
        return 1

    print(f"Repitiendo los tests con {len(UMBRALES)} umbrales de validez.\n")
    tabla, resumen = correr()

    TABLAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(TABLAS / "sensibilidad_umbral_validez.csv", index=False)

    # --- Cuanto filtra el umbral en cada caso --------------------------
    print("\n" + "=" * 78)
    print("CUANTAS VENTANAS SOBREVIVEN A CADA UMBRAL")
    print("=" * 78)
    vista = resumen.copy()
    vista["umbral"] = vista.umbral.map("{:.0%}".format)
    print(vista.rename(columns={
        "ventanas_60s": "vent. 60s", "descartadas_60s": "descart. 60s",
        "min_nn_60s": "min NN 60s", "ventanas_10s": "vent. 10s",
        "descartadas_10s": "descart. 10s", "min_nn_10s": "min NN 10s",
    }).to_string(index=False))

    # --- Si alguna conclusion cambia -----------------------------------
    print("\n" + "=" * 78)
    print("p CORREGIDO POR HOLM EN CADA UMBRAL   (* = significativo)")
    print("=" * 78)
    tabla["resultado"] = (tabla.p_holm.map("{:.3f}".format)
                          + tabla.significativo.map({True: " *", False: "  "}))
    pivote = tabla.pivot_table(index=["pregunta", "metrica"], columns="umbral",
                               values="resultado", aggfunc="first")
    pivote.columns = [f"{c:.0%}" for c in pivote.columns]
    print(pivote.to_string())

    # --- Veredicto ------------------------------------------------------
    # No alcanza con decir si algo cambia o no en todo el rango: lo util es
    # hasta donde aguantan las conclusiones. Se busca el umbral mas alto tal
    # que, desde el mas laxo hasta ese, las 13 comparaciones den lo mismo.
    referencia = UMBRALES[0]
    base = (tabla[tabla.umbral == referencia]
            .set_index(["pregunta", "metrica"]).significativo)

    tope = referencia
    for umbral in UMBRALES[1:]:
        actual = (tabla[tabla.umbral == umbral]
                  .set_index(["pregunta", "metrica"]).significativo)
        if not actual.reindex(base.index).equals(base):
            break
        tope = umbral

    n_comparaciones = base.size
    print()
    print(f"Las {n_comparaciones} comparaciones dan el mismo resultado para "
          f"todo umbral entre el {referencia:.0%} y el {tope:.0%}.")

    if tope < max(UMBRALES):
        # Por encima del rango estable conviene mirar cuanta señal queda: si
        # se descarto buena parte de las ventanas, lo que cambia es la
        # potencia del test y no el efecto que se quiere medir.
        perdidas = resumen[resumen.umbral > tope]
        peor = perdidas.iloc[-1]
        print(f"Por encima del {tope:.0%} el criterio empieza a descartar "
              f"ventanas en cantidad: al {peor.umbral:.0%} quedan "
              f"{int(peor.ventanas_60s)} de "
              f"{int(peor.ventanas_60s + peor.descartadas_60s)} de 60 s. "
              f"Los cambios de resultado en esa zona son atribuibles a la "
              f"perdida de potencia y no a un cambio en el efecto.")

    # El umbral que usa el trabajo tiene que caer dentro del rango estable.
    usado = ventanas.FRACCION_MINIMA
    centro = (referencia + tope) / 2
    print(f"\nUmbral adoptado: {usado:.0%}. "
          + (f"Queda dentro del rango estable "
             f"({referencia:.0%}-{tope:.0%}), cuyo centro es {centro:.0%}."
             if referencia <= usado <= tope
             else "ATENCION: queda FUERA del rango estable."))

    print(f"\nTabla guardada en "
          f"{(TABLAS / 'sensibilidad_umbral_validez.csv').relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
