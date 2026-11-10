# Función del archivo:  corre todo el trabajo en orden del pipeline, de la señal cruda a las tablas y figuras.

# CORRER (desde la raiz del repositorio):
#   python main.py                      -> todo (saltea los pasos lentos si ya estan hechos)
#   python main.py --reprocesar         -> vuelve a detectar los QRS de los 57 registros
#   python main.py --desde estadistica  -> arranca desde ese paso (usa lo que ya esta guardado)
#   python main.py --desde resumen      -> solo imprime las tablas de resultados

# Pasos:
# 1. descarga     : baja el dataset de PhysioNet (solo si no esta en data/raw/)
# 2. procesamiento: filtrado, Pan-Tompkins, serie RR y NN de los 57 registros (tarda varios minutos)
# 3. metricas     : ventanas de 60 y 10 s y metricas de HR/HRV de cada una
# 4. seleccion    : que metricas sirven a 10 s y una representante por familia
# 5. estadistica  : tests de P1, P2, P3a y P3b
# 6. figuras      : figuras del procesamiento y de los resultados
# 7. resumen      : imprime las tablas de resultados de cada pregunta y las
#                   guarda en results/resumen_resultados.txt

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
CRUDO = RAIZ / "data" / "raw"
PROCESADO = RAIZ / "data" / "processed"
CARACTERISTICAS = RAIZ / "data" / "features"
TABLAS = RAIZ / "results" / "tables"
RESUMEN = RAIZ / "results" / "resumen_resultados.txt"
LOG = RAIZ / "results" / "log_ejecucion.txt"     # todo lo que imprimen los scripts

# Cada paso: nombre y scripts que lo componen, en el orden en que se corren.
# El orden importa: cada script lee lo que guardo el anterior.
PASOS = {
    "descarga": ["scripts/descargar_dataset.py"],
    "procesamiento": ["scripts/procesar_registros.py"],
    "metricas": ["modulos/ventanas.py"],
    "seleccion": ["scripts/estabilidad_ventanas.py",    # -> metricas aptas para 10 s
                  "scripts/correlacion_metricas.py"],   # -> familias_metricas.csv (la usa estadistica)
    "estadistica": ["modulos/estadisticas.py"],
    "figuras": ["scripts/generar_figuras.py",
                "scripts/figuras_estadisticas.py"],
    "resumen": [],    # no corre scripts: lo hace imprimir_resumen(), mas abajo
}


# Corre un script con el mismo Python que main.py (asi usa el mismo entorno virtual).
# Lo que imprime el script va al log, no a la pantalla: en pantalla solo queda
# una linea por script y, al final, el resumen de resultados.
# Si el script falla, se corta todo: los pasos siguientes dependen de su salida.
def correr(script, log):
    print(f"   {script} ...", end=" ", flush=True)
    inicio = time.time()

    log.write(f"\n{'=' * 70}\n>> {script}\n{'=' * 70}\n")
    log.flush()

    # MPLBACKEND=Agg: las figuras se guardan en archivo, sin abrir ventanas.
    entorno = {**os.environ, "MPLBACKEND": "Agg"}
    resultado = subprocess.run([sys.executable, script], cwd=RAIZ, env=entorno,
                               stdout=log, stderr=subprocess.STDOUT)

    if resultado.returncode != 0:
        # Se muestran las ultimas lineas del log, que es donde quedo el error.
        log.flush()
        ultimas = LOG.read_text(encoding="utf-8").splitlines()[-20:]
        print(f"FALLO\n\n" + "\n".join(ultimas))
        print(f"\nSe corta la ejecucion. Salida completa en {LOG.relative_to(RAIZ)}")
        sys.exit(resultado.returncode)
    print(f"ok ({time.time() - inicio:.0f} s)")


# ¿Ya esta bajado el dataset? Hace falta la tabla de grupos y al menos un participante.
def hay_dataset():
    return (CRUDO / "subject_groups.csv").exists() and any(CRUDO.glob("VP*"))


def main():
    ap = argparse.ArgumentParser(description="Corre el pipeline completo del TPS.")
    ap.add_argument("--reprocesar", action="store_true",
                    help="volver a detectar los QRS aunque ya esten guardados")
    ap.add_argument("--desde", choices=list(PASOS), default="descarga",
                    help="paso desde el que arrancar (los anteriores se saltean)")
    args = ap.parse_args()

    # Los pasos a correr: desde el pedido hasta el final.
    nombres = list(PASOS)
    pendientes = nombres[nombres.index(args.desde):]

    # El log se reescribe en cada corrida: guarda todo lo que imprimen los scripts.
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as log:
        for paso in pendientes:
            if paso == "resumen":
                continue

            # La descarga solo se hace si falta el dataset (son ~375 MB).
            if paso == "descarga" and hay_dataset():
                print(f"{paso}: el dataset ya esta en data/raw/, se saltea")
                continue

            # La deteccion de QRS es lo mas lento. Si ya esta hecha, se reutiliza,
            # salvo que se pida --reprocesar o que se haya arrancado desde este paso.
            if (paso == "procesamiento" and not args.reprocesar and args.desde != "procesamiento"
                    and (PROCESADO / "intervalos.csv").exists()):
                print(f"{paso}: los QRS ya estan detectados, se saltea (--reprocesar para rehacerlo)")
                continue

            # estadisticas.py guarda las metricas de Group2 y Group4 la primera vez y
            # despues las reutiliza. Si se recalcularon las metricas, esas copias
            # quedarian viejas: se borran para que las vuelva a generar.
            if paso == "estadistica" and "metricas" in pendientes:
                for grupo in ("group2", "group4"):
                    (CARACTERISTICAS / f"metricas_{grupo}.csv").unlink(missing_ok=True)

            print(f"{paso}:")
            for script in PASOS[paso]:
                correr(script, log)

    # El resumen va siempre al final, se haya arrancado desde donde se haya arrancado.
    imprimir_resumen()
    print(f"\nResumen guardado en {RESUMEN.relative_to(RAIZ)}. Tablas completas en "
          f"results/tables/, figuras en results/figures/ y salida de cada script en "
          f"{LOG.relative_to(RAIZ)}")


# RESUMEN DE RESULTADOS
# Lee las tablas que dejaron los scripts y las muestra con pocas columnas y
# nombres claros. Los CSV de results/tables/ guardan todas las columnas.

# Formato de los numeros: p chicos como "< 0.001", IC en una sola columna.
def _p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _si_no(valor):
    return "si" if valor else "no"


# Columnas que comparten todos los tests de Wilcoxon (P1, P2, P3a, P3b):
# cuanto cambio (Hodges-Lehmann), su IC 95 %, el tamaño de efecto r, el p, y
# si la t pareada llega a la misma conclusion.
def _columnas_test(t, con_holm=True):
    columnas = {
        "n": t.n,
        "Cambio (HL)": t.diferencia.map("{:+.2f}".format),
        "IC 95 %": "[" + t.ic95_inf.map("{:.2f}".format) + ", " + t.ic95_sup.map("{:.2f}".format) + "]",
        "r": t.efecto.map("{:+.2f}".format),
        "p": t.p.map(_p),
    }
    if con_holm:
        columnas["p Holm"] = t.p_holm.map(_p)
    columnas["Signif."] = t.significativo.map(_si_no)
    columnas["t coincide"] = t.coinciden.map(_si_no)
    return columnas


def imprimir_resumen():
    lineas = []

    # Cada seccion se imprime y se guarda a la vez, para que el .txt sea igual a la pantalla.
    def seccion(titulo, nota, tabla):
        texto = f"\n{titulo}\n{'-' * len(titulo)}\n"
        if nota:
            texto += f"{nota}\n\n"
        texto += tabla.to_string(index=False) + "\n"
        print(texto)
        lineas.append(texto)

    # Si falta alguna tabla (por ejemplo, no se corrio la estadistica) se avisa en vez de fallar.
    def leer(ruta):
        if not ruta.exists():
            print(f"  (falta {ruta.relative_to(RAIZ)}: no se puede mostrar esta parte)")
            return None
        return pd.read_csv(ruta)

    encabezado = "\n" + "=" * 70 + "\nRESUMEN DE RESULTADOS\n" + "=" * 70
    print(encabezado)
    lineas.append(encabezado + "\n")
    pd.set_option("display.width", 250)

    # --- Procesamiento de la señal: cuanto se descarto ---
    calidad = leer(PROCESADO / "calidad_deteccion.csv")
    if calidad is not None:
        total, descartados = calidad.intervalos.sum(), calidad.descartados.sum()
        problematicos = calidad[calidad.pct_descartado > 5].sort_values("pct_descartado", ascending=False)
        tabla = pd.DataFrame({
            "Participantes": [len(calidad)],
            "Intervalos RR": [total],
            "Descartados": [f"{descartados} ({100 * descartados / total:.2f} %)"],
            "Por rango": [calidad.fuera_de_rango.sum()],
            "Por salto brusco": [calidad.salto_brusco.sum()],
            "HR media [lpm]": [f"{calidad.hr_media_lpm.mean():.1f}"],
            "> 5 % descartado": [", ".join(f"{f.participante} ({f.pct_descartado} %)"
                                           for f in problematicos.itertuples()) or "ninguno"],
        })
        seccion("Procesamiento de la señal (57 registros)",
                "Intervalos descartados: fuera de 300-2000 ms o con un salto > 20 % respecto del anterior.",
                tabla)

    # --- Metricas elegidas ---
    familias = leer(TABLAS / "familias_metricas.csv")
    estabilidad = leer(TABLAS / "estabilidad_ventanas.csv")
    if familias is not None and estabilidad is not None:
        aptas = set(estabilidad[estabilidad.apta_10s == "si"].metrica)
        tabla = (familias.groupby("familia")
                 .apply(lambda g: pd.Series({
                     "Representante": g[g.representante].metrica.iloc[0],
                     "Otras de la familia": ", ".join(g[~g.representante].metrica) or "-"}),
                        include_groups=False)
                 .reset_index(drop=True))
        tabla["Sirve a 10 s"] = tabla.Representante.map(lambda m: _si_no(m in aptas))
        seccion("Metricas: una representante por familia (las que se testean)",
                f"Metricas confiables en ventanas de 10 s: {', '.join(sorted(aptas))}.",
                tabla)

    # --- P1 ---
    t = leer(TABLAS / "p1_exposicion_vs_reposo.csv")
    if t is not None:
        tabla = pd.DataFrame({"Metrica": t.metrica,
                              "Videos (mediana)": t.mediana_exposicion.map("{:.2f}".format),
                              "Reposo (mediana)": t.mediana_reposo.map("{:.2f}".format),
                              **_columnas_test(t)})
        seccion("P1. ¿Difieren la HR y la HRV entre los videos y el reposo? (Group3, ventanas de 60 s)",
                "Cambio = videos - reposo, por persona. Wilcoxon pareado + Holm; la conclusion sale de p Holm.",
                tabla)

    # --- P2 y su complemento ---
    t = leer(TABLAS / "p2_respuesta_inicial.csv")
    if t is not None:
        tabla = pd.DataFrame({"Mediana [lpm]": t.mediana.map("{:+.2f}".format),
                              "Aceleran / frenan": t.aceleran.astype(str) + " / " + t.frenan.astype(str),
                              **_columnas_test(t, con_holm=False)})
        seccion("P2. ¿La HR cambia apenas empieza el video?",
                "Cambio = HR(primeros 10 s del video) - HR(10 s previos). Wilcoxon contra 0.",
                tabla)

    t = leer(TABLAS / "hr_antes_del_video.csv")
    if t is not None:
        tabla = pd.DataFrame({"HR previa (mediana)": t.mediana_previo.map("{:.1f}".format),
                              "HR reposo (mediana)": t.mediana_reposo.map("{:.1f}".format),
                              "Por encima / debajo": t.por_encima.astype(str) + " / " + t.por_debajo.astype(str),
                              **_columnas_test(t, con_holm=False)})
        seccion("P2, complemento (exploratorio): ¿la HR ya esta alta antes del video?",
                "Cambio = HR(10 s previos al video) - HR(reposo), en lpm. Wilcoxon pareado.",
                tabla)

    # --- P3a: pendientes y Friedman ---
    t = leer(TABLAS / "p3a_pendientes.csv")
    if t is not None:
        tabla = pd.DataFrame({"Metrica": t.metrica,
                              "Cambio video 1 a 16 (mediana)": t.mediana_cambio_1_a_16.map("{:+.2f}".format),
                              "Bajan / suben": t.bajan.astype(str) + " / " + t.suben.astype(str),
                              **_columnas_test(t),
                              "Sentido": t.sentido})
        seccion("P3a. ¿La respuesta cambia a lo largo de los 16 videos? (habituacion entre videos)",
                "Cambio = pendiente por video de cada persona. Wilcoxon de las pendientes contra 0 + Holm.",
                tabla)

    t = leer(TABLAS / "p3a_friedman.csv")
    if t is not None:
        tabla = pd.DataFrame({"Metrica": t.metrica, "n": t.n,
                              "Chi2": t.chi2.map("{:.1f}".format),
                              "W de Kendall": t.w_kendall.map("{:.2f}".format),
                              "p": t.p.map(_p), "p Holm": t.p_holm.map(_p),
                              "Signif.": t.significativo.map(_si_no)})
        seccion("P3a, complemento: Friedman sobre las 16 posiciones de video",
                "¿Importa la posicion del video? W de Kendall: 0.1 chico, 0.3 mediano, 0.5 grande.",
                tabla)

    # --- P3b y su replicacion ---
    t = leer(TABLAS / "p3b_dentro_del_video.csv")
    if t is not None:
        tabla = pd.DataFrame({"Metrica": t.metrica,
                              "Primeros 10 s": t.mediana_primera.map("{:.2f}".format),
                              "Ultimos 10 s": t.mediana_ultima.map("{:.2f}".format),
                              "Bajan / suben": t.bajan.astype(str) + " / " + t.suben.astype(str),
                              **_columnas_test(t),
                              "Sentido": t.sentido})
        seccion("P3b. ¿La respuesta cambia dentro de cada video? (Group3)",
                "Cambio = ultimos 10 s - primeros 10 s, promediado sobre los 16 videos. Wilcoxon pareado + Holm.",
                tabla)

    t = leer(TABLAS / "p3b_replicacion.csv")
    if t is not None:
        tabla = pd.DataFrame({"Grupo": t.grupo, "Metrica": t.metrica,
                              "Bajan / suben": t.bajan.astype(str) + " / " + t.suben.astype(str),
                              **_columnas_test(t),
                              "Sentido": t.sentido})
        seccion("P3b, replicacion (exploratoria): la misma comparacion en los tres grupos",
                "Group2 y Group4 hacen biofeedback durante los videos: solo vale la comparacion dentro del video.",
                tabla)

    RESUMEN.parent.mkdir(parents=True, exist_ok=True)
    RESUMEN.write_text("".join(lineas), encoding="utf-8")


if __name__ == "__main__":
    main()
