import numpy as np
import matplotlib.pyplot as plt


def _mostrar_o_guardar(fig, guardar):
    """Muestra la figura en pantalla, o la guarda en disco si se indicó ruta.

    Permite usar las mismas funciones de forma interactiva (en un notebook) y
    para generar las figuras del informe, sin duplicar el código del gráfico.
    """
    if guardar is None:
        plt.show()
    else:
        fig.savefig(guardar, dpi=150, bbox_inches="tight")
        plt.close(fig)


def graficar_ecg(ecg, fs, t_ini=0, t_fin=None, color="tab:blue",
                 picos=None, titulo="ECG", guardar=None):
    """
    ecg    : señal (lista con voltajes en [mV])
    fs     : frecuencia de muestreo [Hz]
    t_ini  : inicio del tramo [s]
    t_fin  : fin del tramo [s] (None = hasta el final)
    color  : color de la señal (nombre, hex o código de matplotlib)
    picos  : (opcional) índices de muestra de las R para marcarlas
    guardar: (opcional) ruta donde guardar la figura. Si es None, se muestra
             en pantalla como hasta ahora.
    """
    ecg = np.asarray(ecg)
    t_total = len(ecg) / fs
    if t_fin is None or t_fin > t_total:
        t_fin = t_total
    if t_ini < 0 or t_ini >= t_fin:
        raise ValueError("Intervalo inválido: se requiere 0 <= t_ini < t_fin")

    i0, i1 = int(round(t_ini * fs)), int(round(t_fin * fs))
    t = np.arange(i0, i1) / fs

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, ecg[i0:i1], color=color, linewidth=1.2)

    if picos is not None:
        picos = np.asarray(picos, dtype=int)
        picos = picos[(picos >= i0) & (picos < i1)]
        ax.plot(picos / fs, ecg[picos], "o", color="blue", label="R")
        ax.legend()

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Amplitud [mV]")
    ax.set_title(titulo)
    ax.set_xlim(t_ini, t_fin)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _mostrar_o_guardar(fig, guardar)

def graficar_tacograma(picos_R, fs, t_ini=0, t_fin=None, color="tab:blue",
                       en_ms=True, titulo="Tacograma", guardar=None):
    """
    picos_R : índices de muestra de las ondas R
    fs      : frecuencia de muestreo [Hz]
    t_ini   : inicio del tramo [s]
    t_fin   : fin del tramo [s] (None = hasta el final)
    color   : color del gráfico
    en_ms   : True = RR en milisegundos, False = en segundos
    guardar : (opcional) ruta donde guardar la figura. Si es None, se muestra
              en pantalla como hasta ahora.
    """
    picos_R = np.sort(np.asarray(picos_R, dtype=int))
    if len(picos_R) < 2:
        raise ValueError("Se necesitan al menos 2 picos R")

    rr = np.diff(picos_R) / fs
    t_rr = picos_R[1:] / fs
    if en_ms:
        rr = rr * 1000

    if t_fin is None:
        t_fin = t_rr[-1]
    m = (t_rr >= t_ini) & (t_rr <= t_fin)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_rr[m], rr[m], color=color, markersize=3, linewidth=1)
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("RR [ms]" if en_ms else "RR [s]")
    ax.set_title(titulo)
    ax.set_xlim(t_ini, t_fin)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _mostrar_o_guardar(fig, guardar)

def graficar_correlacion(validos, corr, fs, t_ini=0, t_fin=None, color="tab:blue",
                         umbral=None, titulo="Correlación con plantilla local",
                         guardar=None):
    """
    validos : índices de muestra de las R (salida de correlacion_latidos)
    corr    : correlación de cada latido con su plantilla
    fs      : frecuencia de muestreo [Hz]
    t_ini   : inicio del tramo [s]
    t_fin   : fin del tramo [s] (None = hasta el final)
    color   : color del gráfico
    umbral  : (opcional) valor de corte. Se dibuja una línea y se marcan en rojo
              los latidos por debajo. Si es "auto", usa mediana - 4*MAD.
    guardar : (opcional) ruta donde guardar la figura. Si es None, se muestra
              en pantalla como hasta ahora.
    """
    validos = np.asarray(validos, dtype=int)
    corr = np.asarray(corr, dtype=float)
    t = validos / fs

    if t_fin is None:
        t_fin = t[-1]
    m = (t >= t_ini) & (t <= t_fin)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t[m], corr[m], "o-", color=color, markersize=3, linewidth=1)

    if umbral is not None:
        if umbral == "auto":
            mad = np.median(np.abs(corr - np.median(corr)))
            umbral = np.median(corr) - 4 * 1.4826 * mad
        ax.axhline(umbral, color="k", linestyle="--", linewidth=1,
                   label=f"umbral = {umbral:.2f}")
        bajo = m & (corr < umbral)
        ax.plot(t[bajo], corr[bajo], "o", color="red", markersize=5,
                label=f"bajo umbral ({bajo.sum()})")
        ax.legend()

    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel("Correlación")
    ax.set_title(titulo)
    ax.set_xlim(t_ini, t_fin)
    ax.set_ylim(min(corr[m].min(), 0.9) - 0.02, 1.01)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _mostrar_o_guardar(fig, guardar)