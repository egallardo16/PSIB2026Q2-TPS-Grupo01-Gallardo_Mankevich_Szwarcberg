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
es una linea de base previa a la exposicion. Ademas, en Group2 y Group4 durante
los videos hacian una tarea de pseudo-biofeedback (tapping) que no hacian en el
reposo, asi que la comparacion videos vs reposo quedaria confundida con la
tarea. Por eso el analisis principal se restringe a Group3, el unico grupo sin
esa tarea.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

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

    La tolerancia es necesaria: inicio_s y fin_s son el resultado de restar
    marcas de tiempo, y esa resta deja errores de redondeo de punto flotante.
    Un video de 60 s exactos puede quedar en 59.999999999999886, y sin
    tolerancia la division entera devolveria cero ventanas y se perderia el
    video entero. Se verifico que eso ocurria con unos pocos videos, y que
    cuales se perdian cambiaba segun la version de pandas instalada: el mismo
    codigo daba resultados distintos en una maquina y en otra.

    Diez milisegundos es una tolerancia segura: los triggers del experimento
    tienen resolucion de un segundo, asi que cualquier diferencia de ese
    orden es error numerico y no una diferencia real de duracion.
    """
    TOLERANCIA_S = 0.01
    duracion_evento = evento.fin_s - evento.inicio_s
    cuantas = int((duracion_evento + TOLERANCIA_S) // duracion_s)
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


def delta_hr(tabla):
    """Cambio de frecuencia cardiaca entre la exposicion y el reposo.

    Un numero por participante: cuantos latidos por minuto mas (o menos)
    tuvo en promedio mientras miraba los videos de arañas, comparado con su
    propio reposo.

        delta_hr > 0  se acelero durante la exposicion
        delta_hr < 0  estuvo mas tranquilo durante la exposicion

    Se calcula sobre las ventanas de 60 s, promediando primero dentro de cada
    condicion. Promediar por condicion y despues restar, en lugar de trabajar
    ventana por ventana, evita que los participantes con mas ventanas validas
    pesen mas que los demas.

    Es el resumen por participante de lo que Nmean expresa ventana a
    ventana, pero no es exactamente su promedio: aca se promedian los
    intervalos y recien despues se convierte a lpm, mientras que Nmean
    convierte cada latido a lpm y promedia eso. Como la conversion
    60000/intervalo no es lineal, las dos cuentas difieren un poco (sobre
    esta cohorte, +1.85 contra +1.81 lpm).

    Se calcula por separado porque es la magnitud que se compara entre
    participantes, y tenerla con nombre propio evita confundir el nivel de
    una ventana con el efecto global de la condicion.
    """
    ventanas = tabla[(tabla.window_duration_s == 60) & tabla.valida]

    filas = []
    for participante, suyas in ventanas.groupby("participant_id"):
        # avNN esta en milisegundos; se pasa a latidos por minuto para que el
        # resultado se lea en la misma unidad que el resto del analisis.
        exposicion = suyas[suyas.condition == "exposicion"]["avNN"].dropna()
        reposo = suyas[suyas.condition == "reposo"]["avNN"].dropna()

        # Sin alguna de las dos condiciones no hay comparacion posible.
        if exposicion.empty or reposo.empty:
            continue

        hr_exposicion = 60_000 / exposicion.mean()
        hr_reposo = 60_000 / reposo.mean()

        filas.append({
            "participant_id": participante,
            "hr_exposicion_lpm": round(hr_exposicion, 2),
            "hr_reposo_lpm": round(hr_reposo, 2),
            "delta_hr_lpm": round(hr_exposicion - hr_reposo, 2),
            "videos": len(exposicion),
        })

    return pd.DataFrame(filas)


def respuesta_inicial(duracion_s=10, participantes=None):
    """Aceleracion o desaceleracion del pulso al empezar cada video.

    Compara la frecuencia cardiaca de los primeros segundos del video contra
    los segundos inmediatamente anteriores, cuando el video todavia no habia
    empezado:

        respuesta = HR(primeros 10 s del video) - HR(10 s previos)

    A diferencia de las demas metricas, esta no puede calcularse desde
    metricas.csv: el periodo previo cae entre dos eventos y no corresponde a
    ninguna ventana de esa tabla. Por eso se vuelve a los intervalos.

    POR QUE 10 SEGUNDOS
    -------------------
    El anteproyecto condiciona esta metrica a que los triggers identifiquen
    un periodo previo comparable. Se verifico que los 288 videos de la
    cohorte tienen al menos 10 s de separacion respecto del evento anterior
    (mediana 21 s), asi que 10 s es la ventana mas larga que se puede usar
    sin que en ningun caso el periodo previo invada el evento anterior.

    Devuelve una fila por video, con el pulso previo, el inicial y su
    diferencia.
    """
    intervalos = pd.read_csv(PROCESADO / "intervalos.csv")
    eventos = pd.read_csv(PROCESADO / "eventos.csv")
    participantes = participantes or cohorte_group3()

    filas = []
    for participante in participantes:
        iv = intervalos[intervalos.participant_id == participante]
        ev = eventos[eventos.participant_id == participante]
        videos = ev[ev.condicion == "exposicion"]

        for _, video in videos.iterrows():
            # Los dos tramos a comparar, a ambos lados del inicio del video.
            nn_previo = serie_en_rango(iv, video.inicio_s - duracion_s,
                                       video.inicio_s)
            nn_inicio = serie_en_rango(iv, video.inicio_s,
                                       video.inicio_s + duracion_s)

            hr_previo = np.nanmean(frecuencia_instantanea(nn_previo))
            hr_inicio = np.nanmean(frecuencia_instantanea(nn_inicio))

            # Si alguno de los dos tramos quedo sin latidos validos, la
            # comparacion no existe y el video se saltea.
            if np.isnan(hr_previo) or np.isnan(hr_inicio):
                continue

            filas.append({
                "participant_id": participante,
                "event_id": video.evento,
                "exposure_order": video.orden,
                "hr_previo_lpm": round(float(hr_previo), 2),
                "hr_inicio_lpm": round(float(hr_inicio), 2),
                "respuesta_lpm": round(float(hr_inicio - hr_previo), 2),
            })

    return pd.DataFrame(filas)


def habituacion(tabla, metricas=None):
    """Como cambia cada metrica a lo largo de los 16 videos, por participante.

    QUE MIDE
    --------
    Es la habituacion cardiaca: si la respuesta a los videos se atenua a
    medida que la persona ve mas. Se calcula ajustando una recta a los 16
    valores de cada metrica en funcion del ORDEN en que la persona vio los
    videos, y quedandose con la pendiente.

        pendiente > 0  la metrica sube video a video
        pendiente ~ 0  no cambia a lo largo de la sesion
        pendiente < 0  baja video a video

    El orden de exposicion sale de exposure_order, que se deduce de los
    tiempos y no del numero de video: la presentacion fue aleatorizada, asi
    que el "CLIP-4" fue el primero para unos y el septimo para otros.

    PARA QUE SIRVE ADEMAS
    ---------------------
    Es el control del principal problema de interpretacion del trabajo. El
    reposo que se usa como referencia esta AL FINAL del experimento, asi que
    si la variabilidad resulta menor durante los videos que en el reposo, hay
    dos explicaciones posibles:

      a) los estimulos reducen la variabilidad  -> lo que se quiere mostrar
      b) la persona se fue relajando con el tiempo, y el reposo, por ser lo
         ultimo, es simplemente el momento mas relajado  -> artefacto

    Las dos predicen cosas distintas para estas pendientes. Si fuera (b), la
    variabilidad deberia subir de forma sostenida del primer al ultimo video.
    Si fuera (a), deberia mantenerse pareja durante los 16 y recien subir en
    el reposo.

    Devuelve una fila por participante y metrica, con la pendiente, el
    cambio total estimado entre el primer y el ultimo video, y el r de la
    regresion.
    """
    metricas = metricas or list(METRICAS)

    # Solo las ventanas de exposicion del analisis principal: la habituacion
    # se define sobre la serie de videos, no sobre el reposo ni la demo.
    videos = tabla[(tabla.condition == "exposicion") &
                   (tabla.window_duration_s == 60) &
                   tabla.valida]

    filas = []
    for participante, suyas in videos.groupby("participant_id"):
        # Ordenadas por el orden real de exposicion de esta persona.
        suyas = suyas.sort_values("exposure_order")

        for metrica in metricas:
            # Se descartan las ventanas donde la metrica no se pudo calcular.
            datos = suyas[["exposure_order", metrica]].dropna()

            # Con menos de la mitad de los videos la recta no es confiable.
            if len(datos) < 8:
                continue

            recta = linregress(datos.exposure_order, datos[metrica])

            filas.append({
                "participant_id": participante,
                "metrica": metrica,
                "videos": len(datos),
                # Cambio de la metrica por cada video que pasa.
                "pendiente": round(recta.slope, 4),
                # Cambio acumulado entre el primer y el ultimo video, que es
                # mas facil de interpretar que la pendiente por video.
                "cambio_total": round(recta.slope * (len(datos) - 1), 3),
                "r": round(recta.rvalue, 3),
            })

    return pd.DataFrame(filas)


if __name__ == "__main__":
    # Este bloque corre solo si se ejecuta el archivo directamente. Genera
    # metricas.csv e imprime un resumen para controlar que las cuentas cierren.

    # El ventaneo necesita las tablas que deja el procesamiento de la señal.
    # Si no estan, se avisa en vez de fallar con un error de archivo.
    if not (PROCESADO / "intervalos.csv").exists():
        print("Falta intervalos.csv. Correr antes: "
              "python scripts/procesar_registros.py")
        sys.exit(1)

    # Sin argumentos, generar() usa la cohorte de Group3.
    tabla = generar()

    # exist_ok evita que falle si la carpeta ya existe de una corrida previa.
    CARACTERISTICAS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CARACTERISTICAS / "metricas.csv", index=False)

    print(f"Participantes : {tabla.participant_id.nunique()}")
    print(f"Ventanas      : {len(tabla)}\n")

    # Resumen por duracion de ventana y condicion. Sirve para verificar a
    # simple vista que la cantidad de ventanas sea la esperada: por ejemplo,
    # 18 participantes x 16 videos = 288 ventanas de 60 s de exposicion, y
    # seis veces mas en las de 10 s.
    resumen = (tabla.groupby(["window_duration_s", "condition"])
               .agg(ventanas=("valida", "size"),      # cuantas se generaron
                    validas=("valida", "sum"),        # cuantas pasaron el minimo
                    nn_mediana=("n_intervalos", "median"))  # latidos por ventana
               .reset_index())
    print(resumen.to_string(index=False))

    # Proporcion de ventanas descartadas por tener muy pocos latidos. Si
    # fuera alta, habria que revisar el umbral o la calidad de la deteccion.
    print(f"\nVentanas no validas: {(~tabla.valida).sum()} de {len(tabla)} "
          f"({100 * (~tabla.valida).sum() / len(tabla):.1f} %)")

    # ------------------------------------------------------------------
    # Metricas derivadas, de un valor por participante
    # ------------------------------------------------------------------
    cambio = delta_hr(tabla)
    inicial = respuesta_inicial()

    # Se promedia la respuesta inicial de los 16 videos de cada persona para
    # tener, igual que delta_hr, un valor por participante.
    resumen_inicial = (inicial.groupby("participant_id")
                       .agg(respuesta_inicial_lpm=("respuesta_lpm", "mean"),
                            videos_con_respuesta=("respuesta_lpm", "size"))
                       .round(2).reset_index())

    derivadas = cambio.merge(resumen_inicial, on="participant_id", how="outer")
    derivadas.to_csv(CARACTERISTICAS / "derivadas.csv", index=False)

    print("\n" + "=" * 62)
    print("METRICAS DERIVADAS (un valor por participante)")
    print("=" * 62)
    print(derivadas.to_string(index=False))

    print(f"\ndelta_hr medio          : {cambio.delta_hr_lpm.mean():+.2f} lpm "
          f"({(cambio.delta_hr_lpm > 0).sum()}/{len(cambio)} se aceleran)")
    print(f"respuesta inicial media : "
          f"{resumen_inicial.respuesta_inicial_lpm.mean():+.2f} lpm")

    # ------------------------------------------------------------------
    # Habituacion: como evoluciona cada metrica a lo largo de los 16 videos
    # ------------------------------------------------------------------
    hab = habituacion(tabla)
    hab.to_csv(CARACTERISTICAS / "habituacion.csv", index=False)

    print("\n" + "=" * 62)
    print("HABITUACION: cambio de cada metrica del primer al ultimo video")
    print("=" * 62)

    # Se prueba si las pendientes de los participantes se apartan de cero de
    # forma sistematica. Si lo hicieran, la comparacion contra el reposo
    # estaria confundida con el paso del tiempo.
    from scipy.stats import wilcoxon

    print(f"{'metrica':<12}{'cambio medio':>14}{'suben':>8}{'p':>10}")
    for metrica, grupo in hab.groupby("metrica"):
        # Wilcoxon de las pendientes contra cero: ¿hay tendencia comun?
        _, p = wilcoxon(grupo.pendiente)
        suben = (grupo.pendiente > 0).sum()
        print(f"{metrica:<12}{grupo.cambio_total.mean():>14.2f}"
              f"{suben:>5}/{len(grupo):<2}{p:>10.4f}")

    print("\nSi una metrica subiera de forma sostenida a lo largo de los 16")
    print("videos, su diferencia contra el reposo final no seria atribuible")
    print("a los estimulos, sino al paso del tiempo.")

    print(f"\nGuardado en data/features/metricas.csv")
    print(f"             data/features/derivadas.csv")
    print(f"             data/features/habituacion.csv")
