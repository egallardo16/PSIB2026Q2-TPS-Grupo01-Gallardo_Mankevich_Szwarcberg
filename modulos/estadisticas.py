# #Función del archivo: realizar los tests que dicen si una diferencia en una métrica es real o casualidad --> permiten sacar conclusiones. 

# ¿QUE HACE? : resume las metricas a un valor por participante y condicion (exposicion o reposo), 
# y aplica el test estadistico que responde cada pregunta.
#  ENTRA: 
# -metricas.csv: una fila por ventana, tiene las 14 métricas calculadas para cada ventana (pedazo de señal). 
# -derivadas.csc: una fila por paciente. Comapra dos momentos disitntos, cada medida es una resta entre dos partes del experimento, que incluye info de distintas ventanas.
        #Ejemplo: deltaHR= HR en exposición a videos - HR en reposo. 
        #Respuesta inicial= pulso en los primeros 10 segundos de video - pulso en los 10 s antes del video. 
# -habituacion.csv: una fila por persona y métrica, se compara la métrica en los distintos videos --> se analiza tendencia a subir o bajar a lo largo del tiempo, relacion con los estímulos. 


# SALE: p1_exposicion_vs_reposo.csv --> contiene una fila por métrica, diciendo 
# cuánto cambió de exposición a reposo, y entre qué valores está probablemente,
# resultados de los distintos test. 

# CORRER: python modulos/estadisticas.py

# Preguntas del trabajo
# P1: ¿Difieren la frecuencia cardiaca y su variabilidad entre la exposicion a los videos y el reposo?
# P2: ¿El pulso cambia apenas empieza cada video? (respuesta inicial)
# P3a:  ¿La respuesta baja a lo largo de los 16 videos? (habituacion)
# P3b:  ¿La respuesta baja dentro de cada video?

# Unidad de analisis: el participante
# Cada participante aporta muchas ventanas (16 videos, mas las del reposo).
# Esas ventanas NO son independientes: son la misma persona. Si se testearan
# como si lo fueran, el n quedaria inflado y saldrian p-valores
# artificialmente chicos (pseudo-replicacion; Hurlbert, 1984).
# Por eso, antes de cualquier test, cada metrica se resume en UN valor por
# participante y condicion: el promedio de sus ventanas validas. Asi el n de
# cada test es la cantidad de participantes, y las comparaciones son
# PAREADAS: cada persona se compara contra si misma.


# Para comparar dos condiciones en las mismas personas hay dos opciones:
#   - Wilcoxon de rangos con signo: no supone nada sobre la forma de los
#     datos. Mira si las diferencias por participante tienden hacia un lado.
#   - t de Student pareada: compara la media de las diferencias y supone que
#     tienen distribucion normal.

# Analisis principal: Wilcoxon, elegido DE ANTEMANO para todas las metricas.
#   - Las metricas de HRV suelen tener distribuciones asimetricas.
# Las conclusiones del trabajo salen de este test.

# Analisis de sensibilidad: t pareada.
#   Se corre siempre, al lado de Wilcoxon, para ver si la conclusion depende
#   del test elegido. 

#     - si los dos tests coinciden, la conclusion no depende de suponer normalidad;
#     - si no coinciden, se discute por que (en general, algun participante
#       extremo que arrastra la media), pero vale lo que dice Wilcoxon.


#Cada test devuelve 3 cosas, cada una responde a una cosa distinta.
#1. Hay diferencia o puede ser por azar? --> p-valor. 
#2. ¿Cuánto cambió? en las unidades de la métrica --> Hodges-Lehmann con su Intervalo de Confianza 95 %.
 # --> IC 95%: si repitieras el estudio muchas veces, el 95 % de los intervalos calculados así contendrían el valor real.
#3. ¿es un cambio chico o grande? en una escala sin unidades --> la r rango-biserial.
 # --> va de -1 a 1; El signo da la dirección. Negativo = la métrica fue menor en exposición que en reposo; positivo = fue mayor.
 #  --> El valor absoluto dice qué tan consistente fue el cambio. Se puede traducir a porcentaje: el peso de los cambios en la dirección dominante es (1 + |r|) / 2
        #Ejemplos: r = −0,8 → 90 % del peso bajando y 10 % subiendo.
 #--> como no tiene unidades, deja comparar entre métricas: por ejemplo, si el efecto de una es más fuerte que el de otra.


#Alfa: umbral. Si p<α , decimos que la diferencia es real. Usamos alfa=0,05 por convención.
# --> en otras palabras: alfa es cuanto riesgo de falso positivo aceptás --> on α = 0,05 aceptás que, si en realidad las arañas no tuvieran ningún efecto, igual en un 5 % de los casos el test diga "hay diferencia" por pura casualidad.



# Cada test con α = 0,05 tiene un 5 % de chance de dar “significativo” por azar, 
# aunque no haya ningún efecto real. Si hacés muchos tests, esas chances se acumulan:
# Si testeás muchas métricas y aparece alguna con p < 0,05, puede ser pura suerte. Holm corrige eso para que la probabilidad de algún falso positivo en todo el grupo de tests siga siendo del 5 %.


#familia de tests: responden a la misma pregunta. 

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Las rutas se arman desde la ubicacion de este archivo para que funcione desde cualquier carpeta.
RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from modulos.metricas import METRICAS

CARACTERISTICAS = RAIZ / "data" / "features"
TABLAS = RAIZ / "results" / "tables"

# Nivel de significancia: por debajo de este p-valor se considera que la
# diferencia no se explica por azar.
ALFA = 0.05


# 1. PREPARAR LOS DATOS: de una fila por ventana a un valor por participante

def cargar_metricas(): #Lee metricas.csv: una fila por ventana, con todas sus metricas.
    return pd.read_csv(CARACTERISTICAS / "metricas.csv")


def cargar_derivadas(): #Lee derivadas.csv: una fila por participante (delta HR y respuesta inicial).
    return pd.read_csv(CARACTERISTICAS / "derivadas.csv")


def cargar_habituacion(): #Lee habituacion.csv: una fila por participante y metrica, con la pendiente a lo largo de los 16 videos.
    return pd.read_csv(CARACTERISTICAS / "habituacion.csv")


def metricas_representantes():
    familias = pd.read_csv(TABLAS / "familias_metricas.csv")
    return familias.loc[familias.representante, "metrica"].tolist()

# Las metricas elegidas como representantes de cada familia. scripts/correlacion_metricas.py agrupo las metricas que miden casi lo
# mismo (por ejemplo SDNN y SD2) y eligio una por familia. Testear solo
# las representantes evita repetir el mismo resultado varias veces y
# reduce la cantidad de comparaciones que hay que corregir.



# Promedia las ventanas validas de cada participante en cada condicion.

#     Parametros
#     metricas   : tabla de metricas.csv (una fila por ventana)
#     duracion_s : 60 (analisis principal) o 10 (secundario)
#     excluir    : participantes a dejar afuera, para los analisis de
#                  sensibilidad (por ejemplo ["VP03"], que tiene el período de reposo incompleto)

#   Devuelve una fila por participante y condicion (exposicion o reposo),
#   con el promedio de cada metrica.


#toma las métricas calculadas para distintas ventanas de una señal
#  y obtiene un único valor promedio de cada métrica por participante y condición.
def resumir_por_participante(metricas, duracion_s=60, excluir=()):
    ventanas = metricas[
        (metricas.window_duration_s == duracion_s)              # largo pedido
        & metricas.valida                                       # con suficientes latidos
        & metricas.condition.isin(["exposicion", "reposo"])     # sin demo ni biofeedback
        & ~metricas.participant_id.isin(excluir)
    ]

    # groupby junta las filas de cada (participante, condicion) y mean las
    # promedia. Los NaN (metricas que no se pudieron calcular) se ignoran.
    return (ventanas
            .groupby(["participant_id", "condition"])[list(METRICAS)]
            .mean()
            .reset_index())


#Esta función toma los valores de una métrica de HRV en dos condiciones (exposición y reposo) 
#y los organiza para poder comparar ambas condiciones dentro de cada participante.
def pares(resumen, metrica, condicion_a="exposicion", condicion_b="reposo"):
    
    # pivot pasa de "una fila por participante y condicion" a "una fila por
    # participante, una columna por condicion".
    tabla = resumen.pivot(index="participant_id", columns="condition",
                          values=metrica)
    tabla = tabla[[condicion_a, condicion_b]].dropna()
    return tabla[condicion_a].to_numpy(), tabla[condicion_b].to_numpy()

# Devuelve dos vectores x (condicion_a) e y (condicion_b) donde x[i] e y[i]
# son de la misma persona. Es lo que necesita un test pareado. Los
# participantes que no tienen alguna de las dos condiciones se descartan,
# # porque no tienen contra que compararse.
    

# 2. HERRAMIENTAS ESTADISTICAS 

def test_w_pareado(x, y):
#Wilconox: compara dos condiciones de HRV medidas en las mismas personas en exposición vs reposo

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y                     # diferencia de cada participante
    n = len(d)

    # Con tan pocas personas el test no tiene sentido.
    if n < 3:
        return {"n": n, "test": "insuficiente", "p": np.nan}

    # Wilcoxon de rangos con signo sobre las diferencias pareadas.
    resultado = stats.wilcoxon(x, y)

    # Tamaño del cambio y su intervalo de confianza.
    diferencia, ic_inferior, ic_superior = _hodges_lehmann(d)

    return {
        "n": n,
        "test": "Wilcoxon",
        "estadistico": resultado.statistic,
        "p": resultado.pvalue,
        "diferencia": diferencia,
        "ic95_inf": ic_inferior,
        "ic95_sup": ic_superior,
        "efecto": _rango_biserial(d),
    }

# Devuelve un diccionario con:

    #     n            cantidad de participantes comparados
    #     test         "Wilcoxon"
    #     estadistico  W de Wilcoxon
    #     p            p-valor del test, sin corregir
    #     diferencia   estimador de Hodges-Lehmann del cambio x - y
    #     ic95_inf, ic95_sup  límites inferior y superior del intervalo de confianza del 95 % del cambio
    #     efecto       r rango-biserial (de -1 a 1), mide la dirección y magnitud del cambio. 

def test_t_pareado(x, y):  # Analisis de SENSIBILIDAD: t de Student pareada.


    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    n = len(d)

    if n < 3:
        return {"t_p": np.nan}

    resultado = stats.ttest_rel(x, y)

    # IC 95 % de la media: media ± t(0.975, n-1) * error estandar
    media = d.mean()
    desvio = d.std(ddof=1)
    margen = stats.t.ppf(0.975, n - 1) * desvio / np.sqrt(n)

    return {
        "t_estadistico": resultado.statistic,
        "t_p": resultado.pvalue,
        "t_diferencia": media,
        "t_ic95_inf": media - margen,
        "t_ic95_sup": media + margen,
        "t_efecto": media / desvio,
    }

  # Devuelve un diccionario con las mismas medidas que test_pareado, todas
    # con el prefijo "t_" para no confundirlas:
    #     t_estadistico  t de Student
    #     t_p            p-valor, sin corregir
    #     t_diferencia   media de las diferencias x - y
    #     t_ic95_inf, t_ic95_sup   intervalo de confianza del 95 % de esa media
    #     t_efecto       d_z de Cohen: la media de las diferencias medida en
    #                    desvios estandar (0.2 chico, 0.5 mediano, 0.8 grande)

def test_vs_cero(x): #responde: “¿estos valores están sistemáticamente por encima o por debajo de 0?”.
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    ceros = np.zeros_like(x)
    return {**test_w_pareado(x, ceros), **test_t_pareado(x, ceros)}


# P2, respuesta inicial: x es, para cada persona, cuánto cambió la HR al empezar el clip. Si los valores tienden a ser positivos, la HR acelera al ver la araña.
# P3a, habituación: x es, para cada persona, la pendiente de la HR a lo largo de los 16 clips. Si las pendientes tienden a ser negativas, la respuesta baja con las exposiciones repetidas.


#Ajusta una lista de p-valores por el metodo de Holm (1979).
    # Como funciona: se ordenan los p de menor a mayor. El mas chico se
    # multiplica por m (la cantidad de tests), el segundo por m-1, el tercero
    # por m-2, y asi. Ademas se exige que los ajustados no bajen al avanzar en
    # el orden, y se topean en 1.

def corregir_holm(pvalores):
    
    p = np.asarray(pvalores, dtype=float)
    m = len(p)
    orden = np.argsort(p)            # posiciones de menor a mayor p

    ajustados = np.empty(m)
    maximo_hasta_aca = 0.0
    for k, i in enumerate(orden):
        valor = min(1.0, (m - k) * p[i])
        maximo_hasta_aca = max(maximo_hasta_aca, valor)
        ajustados[i] = maximo_hasta_aca
    return ajustados


#Calcula el rango bisceral. Se ordenan las diferencias por su valor absoluto

def _rango_biserial(d):
   
    d = d[d != 0]                    # los empates en cero no aportan direccion
    rangos = stats.rankdata(np.abs(d))
    w_positivo = rangos[d > 0].sum()
    w_negativo = rangos[d < 0].sum()
    return (w_positivo - w_negativo) / (w_positivo + w_negativo)

#va de -1 a 1, r=1 significa que todos los participantes subieron; r=-1, que todos bajaorn;
#r=0 que no hay tendencia.


def _hodges_lehmann(d):
    #Cuanto cambio la metrica, con su IC 95 % (Hodges & Lehmann, 1963).

    # Es la medida del cambio que corresponde al test de Wilcoxon, y por eso su
    # intervalo de confianza es coherente con el p-valor: si p < 0.05, el
    # intervalo no incluye el cero.

    # Como se calcula:
    #   1. Se promedian las diferencias de a pares, (d_i + d_j) / 2, para todos
    #      los pares i <= j (incluido cada valor consigo mismo). Se llaman
    #      "promedios de Walsh". Con 18 participantes son 171.
    #   2. El estimador es la mediana de esos promedios.
    #   3. El IC 95 % se lee en esa misma lista ordenada: se descartan los k
    #      valores de cada punta, donde k sale de la distribucion exacta del
    #      estadistico de Wilcoxon (ver _valor_critico_wilcoxon).

    # Devuelve (estimador, ic_inferior, ic_superior).

    n = len(d)
    i, j = np.triu_indices(n)                     # todos los pares i <= j
    walsh = np.sort((d[i] + d[j]) / 2)
    m = len(walsh)                                # n (n + 1) / 2

    k = _valor_critico_wilcoxon(n)
    return float(np.median(walsh)), float(walsh[k]), float(walsh[m - 1 - k])


#Función: calcula cuántos promedios de Walsh hay que descartar en cada punta para armar el IC 95 % de Hodges-Lehmann.
def _valor_critico_wilcoxon(n, alfa=ALFA):
 
    suma_maxima = n * (n + 1) // 2
    # formas[t] = cantidad de combinaciones de signos cuya suma de rangos
    # positivos es t. Se arma agregando los rangos de a uno.
    formas = np.zeros(suma_maxima + 1)
    formas[0] = 1
    for rango in range(1, n + 1):
        formas[rango:] = formas[rango:] + formas[:-rango].copy()

    acumulada = np.cumsum(formas) / 2 ** n         # P(T <= t)
    return int(np.searchsorted(acumulada, alfa / 2, side="right") - 1)



# 3. LAS PREGUNTAS

#p1: ¿Difieren HR y HRV entre exposición y reposo. 
def p1_exposicion_vs_reposo(metricas, lista_metricas=None, duracion_s=60,
                            excluir=()):


    # Para cada metrica: promedio por participante en los videos y en el
    # reposo, Wilcoxon (principal) y t pareada (sensibilidad) entre ambos, y al
    # final la correccion de Holm sobre todas las metricas testeadas, por
    # separado para cada test.

    # Como leer el resultado: si en la exposicion la HR es mayor y rMSSD/SDNN
    # menores que en el reposo, es compatible con mayor activacion durante
    # los videos.

    # Devuelve una fila por metrica, con las medianas de cada condicion y el
    # resultado del test.
    
    lista_metricas = lista_metricas or metricas_representantes()
    resumen = resumir_por_participante(metricas, duracion_s, excluir)

    filas = []
    for metrica in lista_metricas:
        x, y = pares(resumen, metrica)            # x: exposicion, y: reposo
        fila = {
            "metrica": metrica,
            # Descriptivos: para saber hacia donde va la diferencia.
            "mediana_exposicion": np.median(x),
            "mediana_reposo": np.median(y),
        }
        fila.update(test_w_pareado(x, y))       # principal: Wilcoxon
        fila.update(test_t_pareado(x, y))     # sensibilidad: t pareada
        filas.append(fila)

    tabla = pd.DataFrame(filas)

    # Holm, por separado para cada test.
    tabla["p_holm"] = corregir_holm(tabla.p)
    tabla["significativo"] = tabla.p_holm < ALFA
    tabla["t_p_holm"] = corregir_holm(tabla.t_p)
    tabla["t_significativo"] = tabla.t_p_holm < ALFA

    # ¿La conclusion es la misma con los dos tests?
    tabla["coinciden"] = tabla.significativo == tabla.t_significativo
    return tabla


#P2. ¿La sñeal cambia apenas empieza cada video?
# - Dato de entrada: derivadas.csv, columna respuesta_inicial_lpm. Para cada
#     participante es el promedio sobre sus 16 videos de HR(primeros 10 s del video) - HR(10 s previos al video).
#   - Test: test_vs_cero (Wilcoxon contra 0, con la t como sensibilidad).
#     Es un solo test, asi que no hace falta corregir con Holm.
#   - Como leerlo:
#       respuesta > 0 -> el pulso se acelera al aparecer la araña
#                        (respuesta defensiva; Hare & Blevings, 1975)
#       respuesta < 0 -> el pulso se frena (respuesta de orientacion o atencion)
#       p >= 0.05     -> no hay una respuesta inicial consistente entre personas
#   - Limitacion: los triggers tienen resolucion de 1 s y el eje temporal del
#     ECG puede correrse hasta ~3 s respecto del reloj, asi que los dos tramos
#     de 10 s pueden quedar algo corridos respecto del inicio real del video.
#   Devuelve una tabla de una fila con el resultado del test y cuantas
#   personas aceleran y cuantas frenan.
def p2_respuesta_inicial(derivadas):
    respuesta = derivadas["respuesta_inicial_lpm"].dropna()

    fila = {
        "medida": "respuesta_inicial_lpm",
        # Descriptivos: hacia donde va la respuesta y si es pareja entre personas.
        "mediana": respuesta.median(),
        "aceleran": int((respuesta > 0).sum()),
        "frenan": int((respuesta < 0).sum()),
    }
    fila.update(test_vs_cero(respuesta))

    tabla = pd.DataFrame([fila])
    tabla["significativo"] = tabla.p < ALFA
    tabla["t_significativo"] = tabla.t_p < ALFA
    tabla["coinciden"] = tabla.significativo == tabla.t_significativo
    return tabla

# Traduce el signo de un cambio a "mas" o "menos activacion".
# Hace falta porque el signo significa cosas opuestas segun la metrica:
#   - en el pulso (Nmean, avNN), que BAJE es calmarse;
#   - en la variabilidad (SDNN, rMSSD, pNN20, std...), que SUBA es calmarse.
# HRV_ihmig mide tendencia y no tiene una lectura de activacion clara.
# std tampoco: en ventanas de 10 s refleja sobre todo cuanto se MUEVE el pulso
# dentro de la ventana. Al empezar cada video el pulso esta bajando rapido, y
# eso solo ya sube std en la primera ventana, sin que cambie la activacion.
def _sentido(metrica, cambio):
    if metrica in ("HRV_ihmig", "std") or pd.isna(cambio) or cambio == 0:
        return "-"
    if metrica == "avNN":                 # intervalo mas largo = pulso mas lento
        return "menos activacion" if cambio > 0 else "mas activacion"
    if metrica == "Nmean":
        return "menos activacion" if cambio < 0 else "mas activacion"
    return "menos activacion" if cambio > 0 else "mas activacion"


#P3a. ¿La respuesta baja a lo largo de los 16 videos? (habituacion entre videos)
#   - Dato de entrada: habituacion.csv. Para cada participante y metrica tiene
#     la PENDIENTE de la recta "metrica vs orden en que vio los videos (1 a 16)".
#     Pendiente = cuanto cambia la metrica por cada video que pasa.
#   - Test: para cada metrica, test_vs_cero sobre las 18 pendientes.
#     Si las pendientes van de forma sistematica hacia un lado, la respuesta
#     cambia a lo largo de la sesion. Holm sobre las metricas testeadas.
#   - Como leerlo: mirar la columna "sentido".
#       "menos activacion" y significativo -> habituacion (o relajacion con el
#        tiempo: este diseño no permite separar las dos cosas).
#   - Ademas sirve de control de P1: como el reposo es lo ultimo, si la
#     activacion baja video a video, la comparacion videos vs reposo depende
#     de que videos se miren.
#   Devuelve una fila por metrica.
def p3a_habituacion_entre_videos(habituacion, lista_metricas=None):
    lista_metricas = lista_metricas or metricas_representantes()

    filas = []
    for metrica in lista_metricas:
        suyas = habituacion[habituacion.metrica == metrica]
        pendientes = suyas["pendiente"].dropna()

        fila = {
            "metrica": metrica,
            # Descriptivos: cuanto cambia por video y del primero al ultimo.
            "mediana_pendiente": pendientes.median(),
            "mediana_cambio_1_a_16": suyas["cambio_total"].median(),
            "suben": int((pendientes > 0).sum()),
            "bajan": int((pendientes < 0).sum()),
        }
        fila.update(test_vs_cero(pendientes))
        fila["sentido"] = _sentido(metrica, fila["diferencia"])
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    tabla["p_holm"] = corregir_holm(tabla.p)
    tabla["significativo"] = tabla.p_holm < ALFA
    tabla["t_p_holm"] = corregir_holm(tabla.t_p)
    tabla["t_significativo"] = tabla.t_p_holm < ALFA
    tabla["coinciden"] = tabla.significativo == tabla.t_significativo
    return tabla


#P3a, complemento: test de Friedman (Friedman, 1937).
#   - Pregunta: ¿la metrica es distinta segun la posicion del video (1 a 16)?
#     Es como Wilcoxon pero para mas de dos condiciones en las mismas personas:
#     dentro de cada persona ordena sus 16 videos de menor a mayor y mira si
#     algunas posiciones quedan siempre arriba o siempre abajo.
#   - Diferencia con las pendientes: Friedman detecta CUALQUIER diferencia
#     entre posiciones, no solo una tendencia que sube o baja. Por eso se
#     reporta junto con las pendientes, que dicen hacia donde va el cambio.
#   - Necesita a cada persona con los 16 videos validos (ventanas de 60 s).
#   - Tamaño de efecto: W de Kendall (Kendall & Babington Smith, 1939), de 0
#     a 1. Indica que tan parecido es el ordenamiento de las posiciones entre
#     personas: 0 = cada uno distinto, 1 = todos iguales.
#     Orientativo: 0.1 chico, 0.3 mediano, 0.5 grande.
#   Devuelve una fila por metrica.
def p3a_friedman(metricas, lista_metricas=None):
    lista_metricas = lista_metricas or metricas_representantes()

    videos = metricas[(metricas.window_duration_s == 60)
                      & (metricas.condition == "exposicion")
                      & metricas.valida]

    filas = []
    for metrica in lista_metricas:
        # Matriz personas x posiciones: una fila por persona, una columna por
        # posicion del video (1 a 16). Se descartan las personas incompletas.
        matriz = videos.pivot_table(index="participant_id",
                                    columns="exposure_order",
                                    values=metrica).dropna()
        n, k = matriz.shape

        estadistico, p = stats.friedmanchisquare(
            *[matriz[posicion] for posicion in matriz.columns])

        filas.append({
            "metrica": metrica,
            "n": n,
            "posiciones": k,
            "chi2": estadistico,
            "p": p,
            "w_kendall": estadistico / (n * (k - 1)),
        })

    tabla = pd.DataFrame(filas)
    tabla["p_holm"] = corregir_holm(tabla.p)
    tabla["significativo"] = tabla.p_holm < ALFA
    return tabla


# Metricas para P3b. En ventanas de 10 s solo son confiables Nmean, avNN,
# std y rMSSD (scripts/estabilidad_ventanas.py). Se toma una por familia:
#   Nmean (nivel de pulso), std (variabilidad total, ocupa el lugar de SDNN),
#   rMSSD (variabilidad de corto plazo). avNN se deja afuera porque es la
#   misma familia que Nmean.
METRICAS_P3B = ["Nmean", "std", "rMSSD"]


#P3b. ¿La respuesta baja dentro de cada video? (habituacion dentro del estimulo)
#   - Dato de entrada: las ventanas de 10 s de los videos en metricas.csv.
#     Cada video tiene 6: la primera arranca con el video y la ultima cubre
#     sus 10 s finales.
#   - Para cada persona: promedio sobre sus 16 videos de la primera ventana y
#     de la ultima. Asi queda un par de valores por persona.
#   - Test: Wilcoxon pareado (ultima vs primera), t como sensibilidad, Holm
#     sobre las 3 metricas.
#   - Como leerlo: "diferencia" = ultima - primera. Mirar "sentido":
#     "menos activacion" y significativo -> la respuesta baja mientras dura
#     el video (habituacion dentro del estimulo; Matthews et al., 2015).
#   Devuelve una fila por metrica.
def p3b_habituacion_dentro_del_video(metricas, lista_metricas=METRICAS_P3B):
    cortas = metricas[(metricas.window_duration_s == 10)
                      & (metricas.condition == "exposicion")].copy()

    # Posicion de cada ventana dentro de su video: 0 = primera, 5 = ultima.
    # Se calcula ANTES de filtrar las invalidas, para que una ventana
    # descartada no corra la numeracion de las demas.
    por_video = cortas.groupby(["participant_id", "event_id"]).window_start_s
    cortas["posicion"] = por_video.rank(method="first").astype(int) - 1
    cortas["ultima"] = cortas.groupby(["participant_id", "event_id"]).posicion.transform("max")

    cortas = cortas[cortas.valida]
    primeras = (cortas[cortas.posicion == 0]
                .groupby("participant_id")[lista_metricas].mean())
    ultimas = (cortas[cortas.posicion == cortas.ultima]
               .groupby("participant_id")[lista_metricas].mean())

    filas = []
    for metrica in lista_metricas:
        # Se alinean por persona y se descartan las que no tienen las dos.
        par = pd.concat([ultimas[metrica].rename("ultima"),
                         primeras[metrica].rename("primera")], axis=1).dropna()

        fila = {
            "metrica": metrica,
            "mediana_primera": par.primera.median(),
            "mediana_ultima": par.ultima.median(),
            "suben": int((par.ultima > par.primera).sum()),
            "bajan": int((par.ultima < par.primera).sum()),
        }
        fila.update(test_w_pareado(par.ultima, par.primera))   # principal
        fila.update(test_t_pareado(par.ultima, par.primera))   # sensibilidad
        fila["sentido"] = _sentido(metrica, fila["diferencia"])
        filas.append(fila)

    tabla = pd.DataFrame(filas)
    tabla["p_holm"] = corregir_holm(tabla.p)
    tabla["significativo"] = tabla.p_holm < ALFA
    tabla["t_p_holm"] = corregir_holm(tabla.t_p)
    tabla["t_significativo"] = tabla.t_p_holm < ALFA
    tabla["coinciden"] = tabla.significativo == tabla.t_significativo
    return tabla


#Si apretamos el botón para ejecutar, se core esto: 
if __name__ == "__main__":
    if not (CARACTERISTICAS / "metricas.csv").exists():
        print("Falta metricas.csv. Correr antes: python modulos/ventanas.py")
        sys.exit(1)

    metricas = cargar_metricas()
    pd.set_option("display.width", 200)

    # Columnas a mostrar en pantalla (el CSV guarda todas).
    principal = ["metrica", "n", "mediana_exposicion", "mediana_reposo",
                 "diferencia", "ic95_inf", "ic95_sup", "efecto",
                 "p", "p_holm", "significativo"]
    con_t = ["metrica", "t_diferencia", "t_ic95_inf", "t_ic95_sup",
             "t_efecto", "t_p", "t_p_holm", "t_significativo", "coinciden"]

    # --- P1, analisis principal: ventanas de 60 s ---
    p1 = p1_exposicion_vs_reposo(metricas)
    print("P1. Exposicion vs reposo (ventanas de 60 s, una fila por metrica)")
    print("\nPrincipal: Wilcoxon (diferencia = Hodges-Lehmann, efecto = r)\n")
    print(p1[principal].round(4).to_string(index=False))
    print("\nSensibilidad: t pareada (diferencia = media, efecto = d_z)\n")
    print(p1[con_t].round(4).to_string(index=False))

    TABLAS.mkdir(parents=True, exist_ok=True)
    p1.to_csv(TABLAS / "p1_exposicion_vs_reposo.csv", index=False)
    print(f"\nGuardado en results/tables/p1_exposicion_vs_reposo.csv")

    # --- Sensibilidad: sin VP03, que tiene el reposo incompleto (190 de 300 s) ---
    p1_sin_vp03 = p1_exposicion_vs_reposo(metricas, excluir=["VP03"])
    print("\nSensibilidad: P1 sin VP03 (Wilcoxon)\n")
    print(p1_sin_vp03[principal].round(4).to_string(index=False))

    # --- P2: respuesta inicial al empezar cada video ---
    p2 = p2_respuesta_inicial(cargar_derivadas())
    print("\n")
    print("P2. Respuesta inicial: HR(primeros 10 s) - HR(10 s previos), por persona")
    print("    Wilcoxon contra 0 (diferencia = Hodges-Lehmann); t como sensibilidad\n")
    print(p2[["n", "mediana", "aceleran", "frenan", "diferencia", "ic95_inf",
              "ic95_sup", "efecto", "p", "significativo",
              "t_p", "coinciden"]].round(4).to_string(index=False))

    p2.to_csv(TABLAS / "p2_respuesta_inicial.csv", index=False)
    print(f"\nGuardado en results/tables/p2_respuesta_inicial.csv")

    # --- P3a: habituacion a lo largo de los 16 videos ---
    p3a = p3a_habituacion_entre_videos(cargar_habituacion())
    print("\n")
    print("P3a. Habituacion entre videos: pendiente de cada metrica a lo largo de los 16")
    print("     Wilcoxon de las pendientes contra 0 (diferencia = Hodges-Lehmann, por video)\n")
    print(p3a[["metrica", "n", "mediana_cambio_1_a_16", "suben", "bajan",
               "diferencia", "ic95_inf", "ic95_sup", "efecto", "p", "p_holm",
               "significativo", "sentido", "coinciden"]].round(4).to_string(index=False))
    p3a.to_csv(TABLAS / "p3a_pendientes.csv", index=False)

    p3a_f = p3a_friedman(metricas)
    print("\nP3a, complemento: Friedman sobre las 16 posiciones (efecto = W de Kendall)\n")
    print(p3a_f.round(4).to_string(index=False))
    p3a_f.to_csv(TABLAS / "p3a_friedman.csv", index=False)
    print(f"\nGuardado en results/tables/p3a_pendientes.csv y p3a_friedman.csv")

    # --- P3b: habituacion dentro de cada video ---
    p3b = p3b_habituacion_dentro_del_video(metricas)
    print("\n")
    print("P3b. Habituacion dentro del video: ultimos 10 s vs primeros 10 s")
    print("     Wilcoxon pareado (diferencia = ultima - primera, Hodges-Lehmann)\n")
    print(p3b[["metrica", "n", "mediana_primera", "mediana_ultima", "suben",
               "bajan", "diferencia", "ic95_inf", "ic95_sup", "efecto", "p",
               "p_holm", "significativo", "sentido", "coinciden"]].round(4).to_string(index=False))
    p3b.to_csv(TABLAS / "p3b_dentro_del_video.csv", index=False)
    print(f"\nGuardado en results/tables/p3b_dentro_del_video.csv")
