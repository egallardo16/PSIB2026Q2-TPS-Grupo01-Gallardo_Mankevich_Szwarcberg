"""metricas.py -- LOS NUMEROS que describen un tramo de señal.

    QUE HACE : calcula las metricas de frecuencia cardiaca y de variabilidad
               que resumen una ventana de intervalos NN.
    ENTRA    : un vector de intervalos NN en milisegundos
    SALE     : un numero por metrica (o un diccionario con todas)
    SE USA   : sobre cada ventana temporal, para comparar condiciones

--------------------------------------------------------------------------

Las definiciones siguen a Ihmig et al. (2020), "On-line anxiety level
detection from biosignals: machine learning based on a randomized controlled
trial with spider-fearful individuals", PLOS ONE 15(6): e0231517,
ecuaciones 1 a 12. Se indica entre parentesis la ecuacion correspondiente.

Dos convenciones del articulo que conviene tener presentes:

- Las cuatro metricas estadisticas (Nmean, std, NFD, NSD) se calculan sobre la
  señal de FRECUENCIA CARDIACA instantanea, latido a latido, no sobre los
  intervalos. Las siete restantes se calculan sobre los intervalos NN.
- La normalizacion de la ecuacion 1 consiste en restar la frecuencia cardiaca
  media del reposo del propio participante, para poder comparar entre personas
  con pulso basal distinto.

Dos salvedades sobre las formulas publicadas, detectadas al implementarlas:

1. La ecuacion 6, que el articulo llama "HRV", es una suma de diferencias CON
   SIGNO. Telescopa, y queda determinada solo por el primer y el ultimo
   intervalo de la ventana. Ver la funcion `hrv_ihmig`.
2. Las ecuaciones 10 a 12 estan escritas como #(NNI > 50 ms), lo que contaria
   todos los intervalos. El titulo de esas ecuaciones indica la intencion
   correcta, que es contar DIFERENCIAS sucesivas mayores al umbral.
"""

import numpy as np

# Cantidad minima de intervalos para que una metrica tenga sentido. Por debajo
# de este valor las funciones devuelven NaN en lugar de un numero enganoso.
MINIMO_INTERVALOS = 3


def _serie(nn_ms):
    """Deja la entrada como vector de floats, CONSERVANDO los NaN.

    Los intervalos descartados por el control de calidad llegan como NaN. Es
    importante no eliminarlos aca: marcan donde la serie se interrumpe.
    """
    return np.asarray(nn_ms, dtype=float)


def _validos(nn_ms):
    """Solo los intervalos validos, para las metricas que promedian valores."""
    nn = _serie(nn_ms)
    return nn[~np.isnan(nn)]


def _diferencias_sucesivas(nn_ms):
    """Diferencias entre intervalos que eran REALMENTE consecutivos.

    Si entre dos intervalos validos hubo uno descartado, la diferencia entre
    ellos no es una diferencia latido a latido y no debe contarse: restar dos
    latidos que no eran vecinos infla artificialmente rMSSD, pNN50 y NFD.

    Aprovecha que NaN se propaga: al hacer np.diff sobre la serie con huecos,
    toda diferencia que involucre un intervalo descartado queda en NaN y se
    elimina, mientras que las diferencias entre vecinos reales sobreviven.
    """
    diferencias = np.diff(_serie(nn_ms))
    return diferencias[~np.isnan(diferencias)]


def frecuencia_instantanea(nn_ms):
    """Frecuencia cardiaca latido a latido [lpm], conservando los huecos."""
    return 60_000.0 / _serie(nn_ms)


# ---------------------------------------------------------------------------
# Metricas estadisticas, sobre la frecuencia cardiaca instantanea
# ---------------------------------------------------------------------------

def nmean(nn_ms, hr_basal=0.0):
    """Frecuencia cardiaca media normalizada por el reposo. Ecuacion 1.

    Nmean = (1/N) * suma(x_n - mu_t), con mu_t la HR media en reposo.

    Equivale a mean(HR) - hr_basal: cuantos latidos por minuto por encima (o
    por debajo) de su propio reposo estuvo la persona en esta ventana.

    Es la unica metrica de este modulo que depende del basal.
    """
    hr = frecuencia_instantanea(nn_ms)
    if np.sum(~np.isnan(hr)) < 1:
        return np.nan
    return float(np.nanmean(hr) - hr_basal)


def std(nn_ms):
    """Desvio estandar de la frecuencia cardiaca. Ecuacion 3.

    Usa la media de la propia ventana, asi que no depende del basal.
    """
    hr = frecuencia_instantanea(nn_ms)
    if np.sum(~np.isnan(hr)) < 2:
        return np.nan
    return float(np.nanstd(hr, ddof=1))


def nfd(nn_ms):
    """Media de las diferencias absolutas entre latidos consecutivos. Ecuacion 4.

    NFD = 1/(N-1) * suma(|x_n+1 - x_n|) sobre la señal normalizada.

    La normalizacion de la ecuacion 1 resta una constante, que se cancela al
    tomar diferencias, de modo que NFD no depende del basal. Mide que tan
    bruscos son los cambios de un latido al siguiente.
    """
    diferencias = np.diff(frecuencia_instantanea(nn_ms))
    diferencias = diferencias[~np.isnan(diferencias)]
    if len(diferencias) < 1:
        return np.nan
    return float(np.mean(np.abs(diferencias)))


def nsd(nn_ms):
    """Media de las diferencias absolutas a dos latidos de distancia. Ecuacion 5.

    NSD = 1/(N-2) * suma(|x_n+2 - x_n|).

    Ojo: son diferencias con desfasaje de dos posiciones, no la segunda
    derivada discreta. Tampoco depende del basal.
    """
    hr = frecuencia_instantanea(nn_ms)
    if len(hr) < 3:
        return np.nan
    # El NaN de un intervalo descartado invalida las dos diferencias a lag 2
    # que lo involucran, que es justamente lo que se quiere.
    diferencias = hr[2:] - hr[:-2]
    diferencias = diferencias[~np.isnan(diferencias)]
    if len(diferencias) < 1:
        return np.nan
    return float(np.mean(np.abs(diferencias)))


# ---------------------------------------------------------------------------
# Metricas de variabilidad, sobre los intervalos NN
# ---------------------------------------------------------------------------

def hrv_ihmig(nn_ms):
    """Metrica "HRV" tal como la define el articulo. Ecuacion 6.

    HRV = (1/N) * suma_{n=1}^{N-1} (NN_n+1 - NN_n)

    ATENCION: la suma es de diferencias CON SIGNO, de modo que telescopa:

        suma_{n=1}^{N-1} (NN_n+1 - NN_n) = NN_N - NN_1

    y por lo tanto HRV = (NN_N - NN_1) / N. Queda determinada unicamente por
    el primer y el ultimo intervalo de la ventana, descartando todo lo del
    medio: mide deriva, no variabilidad. Dos ventanas con los mismos extremos
    y variabilidad interna opuesta dan el mismo valor.

    Se implementa fiel al articulo para poder reproducirlo y documentar el
    problema. Para analizar variabilidad, usar `rmssd`.
    """
    nn = _validos(nn_ms)
    diferencias = _diferencias_sucesivas(nn_ms)
    if len(nn) < 2 or len(diferencias) < 1:
        return np.nan
    return float(np.sum(diferencias) / len(nn))


def avnn(nn_ms):
    """Duracion media de los intervalos NN [ms]. Ecuacion 7."""
    nn = _validos(nn_ms)
    if len(nn) < 1:
        return np.nan
    return float(np.mean(nn))


def sdnn(nn_ms):
    """Desvio estandar de los intervalos NN [ms]. Ecuacion 8.

    Refleja la variabilidad total, incluidas las oscilaciones lentas, por lo
    que necesita ventanas largas para ser interpretable.
    """
    nn = _validos(nn_ms)
    if len(nn) < 2:
        return np.nan
    return float(np.std(nn, ddof=1))


def rmssd(nn_ms):
    """Raiz del promedio cuadratico de las diferencias sucesivas [ms]. Ecuacion 9.

    rMSSD = sqrt( promedio( (NN_n+1 - NN_n)^2 ) )

    Mide variabilidad de corto plazo: solo compara latidos contiguos, sin
    necesitar que la ventana contenga oscilaciones lentas. Es la metrica de
    variabilidad que la bibliografia senala como mas confiable en ventanas
    cortas.
    """
    diferencias = _diferencias_sucesivas(nn_ms)
    if len(diferencias) < 1:
        return np.nan
    return float(np.sqrt(np.mean(diferencias ** 2)))


def nn50(nn_ms, umbral_ms=50.0):
    """Cantidad de diferencias sucesivas mayores al umbral. Ecuacion 10.

    Es un CONTEO, no un promedio: su resolucion depende de cuantos intervalos
    haya en la ventana.
    """
    diferencias = _diferencias_sucesivas(nn_ms)
    if len(diferencias) < 1:
        return np.nan
    return int(np.sum(np.abs(diferencias) > umbral_ms))


def pnn50(nn_ms):
    """Proporcion de diferencias sucesivas mayores a 50 ms [%]. Ecuacion 11."""
    diferencias = _diferencias_sucesivas(nn_ms)
    if len(diferencias) < 1:
        return np.nan
    return float(100 * nn50(nn_ms) / len(diferencias))


def pnn20(nn_ms):
    """Proporcion de diferencias sucesivas mayores a 20 ms [%]. Ecuacion 12."""
    diferencias = _diferencias_sucesivas(nn_ms)
    if len(diferencias) < 1:
        return np.nan
    return float(100 * nn50(nn_ms, 20.0) / len(diferencias))


# ---------------------------------------------------------------------------
# Metricas adicionales
# ---------------------------------------------------------------------------

def cvnn(nn_ms):
    """Coeficiente de variacion de los NN: sdNN / avNN [%].

    Expresa la variabilidad relativa al nivel medio de cada persona, lo que
    facilita comparar participantes con pulso basal distinto.
    Referencia: Shaffer y Ginsberg (2017), "An Overview of Heart Rate
    Variability Metrics and Norms", Frontiers in Public Health 5:258.
    """
    media = avnn(nn_ms)
    desvio = sdnn(nn_ms)
    if np.isnan(media) or np.isnan(desvio) or media == 0:
        return np.nan
    return float(100 * desvio / media)


def sd1(nn_ms):
    """Dispersion transversal del diagrama de Poincare [ms].

    Describe la variabilidad de corto plazo. Es equivalente a rMSSD salvo un
    factor constante, asi que se comporta igual ante ventanas cortas.
    """
    diferencias = _diferencias_sucesivas(nn_ms)
    if len(diferencias) < 2:
        return np.nan
    return float(np.std(diferencias, ddof=1) / np.sqrt(2))


def sd2(nn_ms):
    """Dispersion longitudinal del diagrama de Poincare [ms].

    SD2 = sqrt(2*sdNN^2 - SD1^2). Describe la variabilidad de largo plazo, de
    modo que hereda de sdNN la necesidad de ventanas largas.
    Referencia: Rahman et al. (2018). Se usa como exploratoria: el propio
    trabajo cuestiona interpretarla como indicador especifico de regulacion
    simpatica.
    """
    nn = _validos(nn_ms)
    if len(nn) < MINIMO_INTERVALOS:
        return np.nan
    desvio, transversal = sdnn(nn_ms), sd1(nn_ms)
    if np.isnan(desvio) or np.isnan(transversal):
        return np.nan
    varianza = 2 * desvio ** 2 - transversal ** 2
    return float(np.sqrt(max(varianza, 0.0)))


# ---------------------------------------------------------------------------
# Calculo de todas las metricas de una ventana
# ---------------------------------------------------------------------------

# Nombre de cada metrica y funcion que la calcula. Se usa para recorrerlas
# todas sin repetir la lista en cada script.
METRICAS = {
    "Nmean": nmean,
    "std": std,
    "NFD": nfd,
    "NSD": nsd,
    "HRV_ihmig": hrv_ihmig,
    "avNN": avnn,
    "SDNN": sdnn,
    "rMSSD": rmssd,
    "NN50": nn50,
    "pNN50": pnn50,
    "pNN20": pnn20,
    "CVNN": cvnn,
    "SD1": sd1,
    "SD2": sd2,
}


def calcular_todas(nn_ms, hr_basal=0.0):
    """Devuelve un diccionario con todas las metricas de una ventana.

    hr_basal se usa unicamente para Nmean; el resto no depende de el.
    """
    resultado = {}
    for nombre, funcion in METRICAS.items():
        resultado[nombre] = (funcion(nn_ms, hr_basal) if nombre == "Nmean"
                             else funcion(nn_ms))
    resultado["n_intervalos"] = int(len(_validos(nn_ms)))
    return resultado


if __name__ == "__main__":
    # Ejemplo con una serie corta, para ver que devuelve cada metrica y
    # comprobar a mano que las cuentas dan.
    ejemplo = np.array([800.0, 900.0, 840.0, 880.0, 850.0])
    print("Serie NN de ejemplo [ms]:", ejemplo.tolist())
    print("Diferencias sucesivas  :", np.diff(ejemplo).tolist())
    print()
    for nombre, valor in calcular_todas(ejemplo).items():
        print(f"  {nombre:<14} {valor:.2f}" if isinstance(valor, float)
              else f"  {nombre:<14} {valor}")
    print()
    print("Comprobacion del telescopeo de la ecuacion 6:")
    print(f"  HRV calculado        = {hrv_ihmig(ejemplo):.2f}")
    print(f"  (NN_ultimo - NN_1)/N = {(ejemplo[-1] - ejemplo[0]) / len(ejemplo):.2f}")
