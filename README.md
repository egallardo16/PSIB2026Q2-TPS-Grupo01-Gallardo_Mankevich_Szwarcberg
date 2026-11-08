# TPS — Procesamiento de señales de ECG

Trabajo práctico de señales de **16.63 Procesamiento de Señales e Imágenes
Biomédicas** (ITBA, 2026 Q2) — Grupo 1.

| Legajo | Integrante | Mail |
|--------|------------|------|
| 65318 | Gallardo, Emanuel | egallardo@itba.edu.ar |
| 65287 | Mankevich, Julieta | jmankevich@itba.edu.ar |
| 65057 | Szwarcberg, Milena | mszwarcberg@itba.edu.ar |

## Objetivo

Desarrollar un algoritmo de procesamiento de señales de electrocardiograma
registradas durante la exposición a estímulos de miedo, para analizar las
variaciones de la actividad cardiovascular asociadas a distintos estados de
activación fisiológica. El trabajo abarca el acondicionamiento de las señales,
la detección de complejos QRS y la extracción de características de frecuencia
cardíaca y su variabilidad (HRV).

## Dataset

[Electrocardiogram, skin conductance and respiration from spider-fearful
individuals watching spider video clips](https://physionet.org/content/ecg-spider-clip/1.0.0/),
versión 1.0.0, publicado en PhysioNet por Ihmig et al. (2020).

- **57 participantes** con registros disponibles (de 80 en el estudio original).
- Señales adquiridas con **BITalino (r)evolution BLE**, a **100 Hz** y 10 bits.
- Cada participante vio un video de demostración, **16 videos de arañas** de
  ~60 s cada uno, y una etapa final de reposo de 300 s.
- Artículo de referencia: *On-line anxiety level detection from biosignals:
  machine learning based on a randomized controlled trial with spider-fearful
  individuals*. PLOS ONE 15(6): e0231517.
  DOI: [10.1371/journal.pone.0231517](https://doi.org/10.1371/journal.pone.0231517)

## Instalación

Requiere **Python 3.10 o superior** (desarrollado con 3.12.9).

```bash
git clone https://github.com/egallardo16/TPS_ECG_GRUPO1.git
cd TPS_ECG_GRUPO1

# Entorno virtual (recomendado, fuera del repositorio)
python3 -m venv ~/.venvs/psib-tps
~/.venvs/psib-tps/bin/pip install -r requirements.txt
```

Para usar el entorno en Jupyter, registrarlo como kernel:

```bash
~/.venvs/psib-tps/bin/python -m ipykernel install --user \
    --name psib-tps --display-name "Python (psib-tps)"
```

## Datos

El dataset completo pesa unos **375 MB**, por lo que **no se versiona en el
repositorio**. Se descarga desde PhysioNet con:

```bash
# todo el dataset (57 participantes)
~/.venvs/psib-tps/bin/python scripts/descargar_dataset.py

# solo las marcas temporales de los eventos (24 KB)
~/.venvs/psib-tps/bin/python scripts/descargar_dataset.py --solo-triggers

# solo algunos participantes
~/.venvs/psib-tps/bin/python scripts/descargar_dataset.py --sujetos VP03 VP06
```

El script deja los archivos en `data/raw/VPxx/`, respetando la estructura de
PhysioNet. Se puede correr varias veces: saltea lo que ya está descargado, así
que también sirve para reintentar una descarga interrumpida.

**No hace falta descargar nada para abrir `Exploratorio.ipynb`**: los cinco
registros de `registros/` están incluidos en el repositorio como muestra.

> **Si la descarga falla**, bajar el ZIP a mano desde la
> [página del dataset](https://physionet.org/content/ecg-spider-clip/1.0.0/)
> y descomprimirlo en `data/raw/`, de modo que quede
> `data/raw/VP03/BitalinoECG.txt`.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── Exploratorio.ipynb          # exploración del dataset y de la estructura de los registros
├── modulos/
│   ├── preprocesamiento.py     # carga de los registros y filtro pasabanda
│   ├── procesamiento.py        # Pan-Tompkins, detección de QRS, serie RR, latidos ectópicos
│   └── visualizacion.py        # gráficos de ECG, tacograma y correlación de latidos
├── probar_funciones.py         # script de prueba manual de las funciones de modulos/
├── scripts/
│   └── descargar_dataset.py    # descarga del dataset desde PhysioNet
├── registros/                  # 5 registros de muestra, versionados
└── data/raw/                   # dataset completo (no versionado, se descarga)
```

## Cómo correr

**Notebook exploratorio.** Abrir `Exploratorio.ipynb` y seleccionar el kernel
`Python (psib-tps)`. Funciona con los registros de muestra, sin descargar nada.

**Pipeline de procesamiento.** Desde la raíz del repositorio:

```bash
~/.venvs/psib-tps/bin/python probar_funciones.py
```

Carga un registro, le aplica el filtro pasabanda, detecta los complejos QRS con
Pan-Tompkins y grafica la señal cruda, la filtrada, la procesada, el tacograma y
la correlación de cada latido con su plantilla local.

> Las rutas de `modulos/preprocesamiento.py` son relativas, así que los scripts
> se tienen que ejecutar **desde la raíz del repositorio**.

## Notas sobre los datos

- La frecuencia de muestreo efectiva (muestras sobre duración según los
  timestamps) es de 99.9–100.0 Hz, por lo que usar la nominal de 100 Hz para
  reconstruir el eje temporal es válido.
- Los 16 videos de arañas están presentes en los 57 participantes, con
  duraciones de 60–61 s.
- El orden de presentación de los videos fue **aleatorizado por participante**,
  así que el orden de exposición se deriva de los timestamps de `Triggers.txt`
  y no del número de video.
- El reposo final (`BIOFEEDBACK-REST`, 300 s) está presente en 56 de los 57
  participantes: **VP26 no lo tiene**. En VP03 la grabación corta antes de que
  termine, quedando 190 s de los 300.
- De los tres grupos del estudio, solo **Group3** no recibió entrenamiento de
  biofeedback respiratorio antes del reposo. En Group2 y Group4, 38 de 39
  participantes tienen un evento de biofeedback inmediatamente previo, lo que
  hay que tener en cuenta si se usa el reposo como línea de base.
