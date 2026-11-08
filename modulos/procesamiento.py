import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans

def panTompkins(fs, ecg):

    f1, f2, orden = 5, 15, 3

    # Pasabanda Butterworth
    sos = butter(N=orden, Wn=[f1, f2], btype="bandpass", fs=fs, output="sos")
    ecg_filtrado = sosfiltfilt(sos, ecg) #filtro hacia adelante y hacia atras

    # Derivador centrado (sin retardo)
    b_derivador = (fs / 8) * np.array([1, 2, 0, -2, -1])
    ecg_derivado = np.convolve(ecg_filtrado, b_derivador, mode="same")

    # Cuadrado
    ecg_cuadrado = ecg_derivado**2

    # Media móvil centrada, con N impar para que el retardo sea entero
    N = int(round(0.15 * fs))
    if N % 2 == 0:
        N += 1
    b_media = np.ones(N) / N
    ecg_integrado = np.convolve(ecg_cuadrado, b_media, mode="same")

    return ecg_integrado, ecg_filtrado

def detectar_qrs(fs, ecg_integrado, ecg_filtrado):
  #umbralizado adaptativo de pan-tompkins (con los coeficientes del paper)
    refractario = int(round(0.200 * fs))
    ventana_T   = int(round(0.360 * fs))
    w_pend      = int(round(0.075 * fs))

    def pendiente(p):
        a, b = max(0, p - w_pend), min(len(ecg_filtrado), p + w_pend)
        return np.max(np.abs(np.diff(ecg_filtrado[a:b])))

    # Candidatos
    picos, _ = find_peaks(ecg_integrado, distance=refractario)

    # Inicialización con los primeros 2 s
    n_ini = int(2 * fs)
    spki = 0.25 * np.max(ecg_integrado[:n_ini])
    npki = 0.50 * np.mean(ecg_integrado[:n_ini])

    qrs, rr = [], []

    for k, p in enumerate(picos):
        valor = ecg_integrado[p]
        thr1 = npki + 0.25 * (spki - npki)

        # Búsqueda hacia atrás
        if qrs and rr and (p - qrs[-1]) > 1.66 * np.mean(rr):
            thr2 = 0.5 * thr1
            previos = [q for q in picos[:k]
                       if q > qrs[-1] + refractario and ecg_integrado[q] > thr2]
            if previos:
                q = max(previos, key=lambda j: ecg_integrado[j])
                rr = (rr + [q - qrs[-1]])[-8:]
                qrs.append(q)
                spki = 0.25 * ecg_integrado[q] + 0.75 * spki
                thr1 = npki + 0.25 * (spki - npki)

        if valor > thr1:
            # Discriminación de onda T
            es_T = bool(qrs) and (p - qrs[-1]) < ventana_T \
                   and pendiente(p) < 0.5 * pendiente(qrs[-1])
            if es_T:
                npki = 0.125 * valor + 0.875 * npki
            else:
                if qrs:
                    rr = (rr + [p - qrs[-1]])[-8:]
                qrs.append(p)
                spki = 0.125 * valor + 0.875 * spki
        else:
            npki = 0.125 * valor + 0.875 * npki

    # Refinamiento: la R es el máximo de |señal filtrada| cerca del pico integrado
    w = int(round(0.100 * fs))
    picos_R = []
    for q in qrs:
        a, b = max(0, q - w), min(len(ecg_filtrado), q + w)
        picos_R.append(a + np.argmax(np.abs(ecg_filtrado[a:b])))

    return np.array(picos_R)

def calcular_serie_rr(picosR, fs):
    
    picosR = np.sort(np.asarray(picosR, dtype=int)) #convierte a array de np y ordena
    intervalos = np.diff(picosR) / fs #calcula la diferencia y lo pasa a segundos
    
    return intervalos


def clasificar_ectopicos(picos_R, fs, ventana=11, umbral_corto=0.85, umbral_largo=1.15):
    """
    picos_R : índices de muestra de las ondas R
    fs      : frecuencia de muestreo [Hz]
    ventana : largo (en latidos) de la mediana local para normalizar los RR

    Devuelve:
      ectopicos : índices de muestra de los latidos clasificados como ectópicos
      etiquetas : vector con 1 = ectópico, 0 = normal (uno por latido analizado)
      latidos   : índices de muestra de los latidos analizados (picos_R[1:-1])
    """
    rr = calcular_serie_rr(picos_R, fs)

    # RR normalizado por el ritmo local (comparable entre estrés y relajación)
    # Esta línea estaba comentada y el K-means se hacía sobre los RR absolutos,
    # en segundos. Como umbral_corto y umbral_largo son proporciones respecto
    # del ritmo local (0.85 y 1.15), y los RR rondan los 0.8 s, la validación
    # de más abajo nunca se cumplía y la función devolvía siempre 0 ectópicos.
    # Al normalizar, los valores quedan alrededor de 1 y los umbrales aplican.
    rr_n = rr / median_filter(rr, size=ventana, mode="nearest")

    # Para cada latido: RR normalizado que lo precede y RR normalizado que lo sigue.
    # Un latido ectópico aparece antes de tiempo (RR previo corto) y es seguido
    # por una pausa compensatoria (RR posterior largo), así que en este plano
    # los ectópicos caen en una zona distinta a la de los latidos normales.
    X = np.column_stack([rr_n[:-1], rr_n[1:]])
    latidos = picos_R[1:-1]

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)

    # El cluster ectópico es el de centroide con RR previo corto y RR posterior largo
    c = km.cluster_centers_
    k_ect = np.argmin(c[:, 0] - c[:, 1])

    # Validación: si el centroide no tiene el patrón corto-largo, no hay ectópicos
    if not (c[k_ect, 0] < umbral_corto and c[k_ect, 1] > umbral_largo):
        etiquetas = np.zeros(len(X), dtype=int)
    else:
        etiquetas = (km.labels_ == k_ect).astype(int)

    return latidos[etiquetas == 1], etiquetas, latidos

def correlacion_latidos(ecg, picos_R, fs, pre=0.25, post=0.45, vecinos=15):

    a, b = int(pre * fs), int(post * fs)
    validos = picos_R[(picos_R >= a) & (picos_R < len(ecg) - b)] 
    M = np.array([ecg[p - a:p + b] for p in validos])

    corr = np.empty(len(validos))
    for i in range(len(validos)):
        j0, j1 = max(0, i - vecinos), min(len(validos), i + vecinos + 1)
        plantilla = np.median(M[j0:j1], axis=0)
        corr[i] = np.corrcoef(M[i], plantilla)[0, 1]
    return validos, corr