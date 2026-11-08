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
la detección de complejos QRS, la extracción de características de frecuencia
cardíaca y su variabilidad (HRV), y su comparación entre condiciones con tests
estadísticos.

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
├── report.md                   # decisiones, criterios y resultados del análisis estadístico
├── requirements.txt
├── Exploratorio.ipynb          # exploración del dataset y de la estructura de los registros
├── modulos/
│   ├── preprocesamiento.py     # carga de los registros y filtro pasabanda
│   ├── procesamiento.py        # Pan-Tompkins, detección de QRS, serie RR, latidos ectópicos
│   ├── eventos.py              # marcas temporales del protocolo, desde Triggers.txt
│   ├── intervalos.py           # serie NN: criterios de validez y descarte de artefactos
│   ├── metricas.py             # métricas de HR y HRV de una ventana (Ihmig et al., 2020 y otras)
│   ├── ventanas.py             # ventanas de 60 y 10 s por evento, ΔHR, respuesta inicial y habituación
│   ├── estadisticas.py         # tests estadísticos de las preguntas P1 a P3b
│   └── visualizacion.py        # gráficos de ECG, tacograma y correlación de latidos
├── probar_funciones.py         # script de prueba manual de las funciones de modulos/
├── scripts/
│   ├── descargar_dataset.py    # descarga del dataset desde PhysioNet
│   ├── procesar_registros.py   # pipeline completo sobre los 57 registros
│   ├── estabilidad_ventanas.py # qué métricas son confiables en ventanas de 10 s
│   ├── correlacion_metricas.py # agrupa las métricas en familias y elige una representante
│   ├── generar_figuras.py      # figuras del procesamiento y de la habituación
│   └── figuras_estadisticas.py # figuras de los resultados estadísticos
├── registros/                  # 5 registros de muestra, versionados
├── results/
│   ├── figures/                # figuras generadas, versionadas
│   └── tables/                 # tablas de métricas y de resultados de los tests, versionadas
└── data/                       # datos crudos y procesados (no versionado)
    ├── raw/                    # dataset descargado de PhysioNet
    ├── processed/              # salidas de procesar_registros.py
    └── features/               # salidas de ventanas.py (métricas por ventana y por persona)
```

Cada módulo y script arranca con un encabezado que indica qué hace, qué recibe
y qué produce.

## Cómo correr

Todo se ejecuta **desde la raíz del repositorio**, porque las rutas son
relativas a ella.

**1. Procesar los registros.** Es el paso principal: carga cada ECG, lo filtra,
detecta los complejos QRS, arma la serie RR y la convierte en serie NN.

```bash
~/.venvs/psib-tps/bin/python scripts/procesar_registros.py

# o un subconjunto, para probar
~/.venvs/psib-tps/bin/python scripts/procesar_registros.py --sujetos VP03 VP06
```

Deja en `data/processed/`:

| Archivo | Contenido |
|---------|-----------|
| `intervalos.csv` | series RR y NN de cada participante, con marca de calidad por intervalo |
| `eventos.csv` | eventos del protocolo, en segundos desde el inicio de cada registro |
| `calidad_deteccion.csv` | indicadores de la detección por participante |
| `picos/VPxx.npy` | posiciones de las ondas R, para no repetir la detección |

**2. Calcular las métricas.** Corta la serie NN en ventanas de 60 y 10 s según
los eventos y calcula las métricas de HR y HRV de cada una. Usa los
participantes de Group3 (ver [Análisis estadístico](#análisis-estadístico)).

```bash
~/.venvs/psib-tps/bin/python modulos/ventanas.py
```

Deja en `data/features/`:

| Archivo | Contenido |
|---------|-----------|
| `metricas.csv` | una fila por ventana, con las 14 métricas y si la ventana es válida |
| `derivadas.csv` | una fila por participante: ΔHR y respuesta inicial |
| `habituacion.csv` | una fila por participante y métrica: pendiente a lo largo de los 16 videos |

**3. Elegir las métricas** (opcional: las tablas ya están en `results/tables/`).
Analiza qué métricas son confiables en 10 s y agrupa las que miden lo mismo:

```bash
~/.venvs/psib-tps/bin/python scripts/estabilidad_ventanas.py
~/.venvs/psib-tps/bin/python scripts/correlacion_metricas.py
```

**4. Correr los tests estadísticos.** Imprime los resultados de cada pregunta y
los guarda en `results/tables/`:

```bash
~/.venvs/psib-tps/bin/python modulos/estadisticas.py
```

| Archivo | Pregunta |
|---------|----------|
| `p1_exposicion_vs_reposo.csv` | P1: videos vs reposo |
| `p2_respuesta_inicial.csv` | P2: respuesta al empezar el video |
| `p3a_pendientes.csv`, `p3a_friedman.csv` | P3a: habituación a lo largo de los 16 videos |
| `p3b_dentro_del_video.csv` | P3b: habituación dentro de cada video |

**5. Generar las figuras** en `results/figures/`:

```bash
~/.venvs/psib-tps/bin/python scripts/generar_figuras.py        # 01 a 05 y 08
~/.venvs/psib-tps/bin/python scripts/figuras_estadisticas.py   # 09 a 11
```

| Figura | Contenido |
|--------|-----------|
| 01–05 | procesamiento de la señal: filtrado, QRS, tacogramas, artefactos, calidad |
| 06–07 | estabilidad de las métricas en 10 s y familias de métricas (las genera el paso 3) |
| 08 | HR y rMSSD a lo largo de los 16 videos, por participante |
| 09 | P1: videos vs reposo, una línea por participante |
| 10 | P3a: HR y rMSSD a lo largo de la sesión, con el nivel del reposo |
| 11 | P2 y P3b: HR antes y durante el video |

**Inspeccionar un participante.** Los módulos se pueden ejecutar sueltos para
ver qué producen:

```bash
~/.venvs/psib-tps/bin/python modulos/eventos.py VP03      # eventos del protocolo
~/.venvs/psib-tps/bin/python modulos/intervalos.py VP12   # intervalos y descartes
```

**Notebook exploratorio.** Abrir `Exploratorio.ipynb` y seleccionar el kernel
`Python (psib-tps)`. Funciona con los registros de muestra, sin descargar nada.

**Pruebas manuales de Emanuel.** `probar_funciones.py` carga un registro de
`registros/`, lo procesa y grafica la señal cruda, la filtrada, la procesada, el
tacograma y la correlación de cada latido con su plantilla local.

## Procesamiento de la señal

1. **Filtrado.** Pasabanda Butterworth de orden 3 entre 0,5 y 40 Hz, aplicado
   hacia adelante y hacia atrás para no introducir desfasaje.
2. **Detección de QRS.** Pan-Tompkins: pasabanda 5–15 Hz, derivador, cuadrado,
   media móvil de 150 ms y umbralizado adaptativo con discriminación de onda T.
3. **Serie NN.** Se descartan los intervalos que no pueden corresponder a
   latidos reales, según dos criterios tomados de la bibliografía:
   - rango fisiológico de 300 a 2000 ms, equivalente a 30–200 lpm
     ([Task Force ESC/NASPE, 1996](https://doi.org/10.1161/01.CIR.93.5.1043),
     *Circulation* 93:1043–1065);
   - variación máxima del 20 % respecto del intervalo anterior (criterio de Malik).

   Los intervalos inválidos se **descartan y no se interpolan**: interpolar
   introduce valores intermedios que reducen artificialmente las métricas
   basadas en diferencias sucesivas, como rMSSD y pNN50.

### Resultados del procesamiento

Sobre los 57 participantes: **182 496 intervalos**, de los cuales se descarta el
**2,36 %** (1122 por rango fisiológico y 3194 por salto brusco).

- **No se detectaron latidos ectópicos** en ninguno de los registros, lo que es
  esperable en una muestra de adultos jóvenes sanos. El control se realizó con
  agrupamiento K-means sobre los intervalos RR normalizados por el ritmo local.
- La principal fuente de intervalos inválidos resultaron ser **falsos positivos
  de detección**, en su mayoría artefactos de movimiento que el detector marca
  como complejos QRS. La figura `04_falso_positivo.png` muestra un caso en el
  que un intervalo de unos 750 ms queda partido en dos de 490 y 260 ms.
- Solo dos participantes superan el 5 % de intervalos descartados: **VP70**
  (59,7 %) y **VP23** (8,3 %).

## Análisis estadístico

El detalle de cada decisión, con su justificación y referencias, está en
[`report.md`](report.md).

- **Participantes:** solo **Group3** (n = 18). En Group2 y Group4 el reposo,
  que es la referencia, viene después de un entrenamiento de respiración lenta
  que modifica la HRV.
- **Métricas:** de las 14 calculadas se testean 5 representantes, una por
  familia de métricas que miden lo mismo: Nmean (HR), SDNN, rMSSD, pNN20 (HRV)
  y HRV de Ihmig (tendencia).
- **Unidad de análisis:** un valor por participante y condición (promedio de
  sus ventanas válidas), para no tratar como independientes las ventanas de una
  misma persona.
- **Tests:** Wilcoxon de rangos con signo como test principal, elegido de
  antemano; t pareada como análisis de sensibilidad; Hodges-Lehmann para el
  tamaño del cambio con su IC 95 %; r rango-biserial como tamaño de efecto;
  Friedman con W de Kendall para las 16 posiciones de video; corrección de Holm
  en cada familia de tests (α = 0,05).
- **No se hace análisis en frecuencia (Lomb):** las ventanas de 10 y 60 s son
  demasiado cortas para estimar LF y HF, y no hay 5 minutos seguidos de
  exposición bajo una misma condición.

| Pregunta | Resultado |
|----------|-----------|
| P1. ¿Difieren HR y HRV entre videos y reposo? | Dirección esperada (HR más alta, HRV más baja), pero no significativa tras Holm: el efecto se diluye al promediar los 16 videos |
| P2. ¿Cambia la HR al empezar el video? | No: la HR ya estaba elevada antes de que empiece |
| P3a. ¿Cambia la respuesta a lo largo de los 16 videos? | Sí: la HR baja y la HRV sube (p Holm = 0,006), concentrado en el paso de la primera a la segunda mitad |
| P3b. ¿Cambia la respuesta dentro de cada video? | Sí: la HR baja en los primeros ~20 s, en los 18 participantes (p Holm < 0,001) |

## Notas sobre los datos

- La frecuencia de muestreo efectiva (muestras sobre duración según los
  timestamps) es de 99.9–100.0 Hz, por lo que usar la nominal de 100 Hz para
  reconstruir el eje temporal es válido.
- Los 16 videos de arañas están presentes en los 57 participantes, con
  duraciones de 60–61 s.
- El orden de presentación de los videos fue **aleatorizado por participante**,
  así que el orden de exposición se deriva de los timestamps de `Triggers.txt`
  y no del número de video. La aleatorización es solo **dentro de dos mitades
  fijas**: las posiciones 1 a 8 siempre corresponden a los clips 1 a 8, y las
  9 a 16 a los clips 9 a 16.
- El reposo final (`BIOFEEDBACK-REST`, 300 s) está presente en 56 de los 57
  participantes: **VP26 no lo tiene**. En VP03 la grabación corta antes de que
  termine, quedando 190 s de los 300.
- De los tres grupos del estudio, solo **Group3** no recibió entrenamiento de
  biofeedback respiratorio antes del reposo. En Group2 y Group4, 38 de 39
  participantes tienen un evento de biofeedback inmediatamente previo, lo que
  hay que tener en cuenta si se usa el reposo como línea de base.
- Group3 es además el grupo con mejor calidad de señal: descarta en promedio el
  0,85 % de los intervalos y ninguno de sus 18 participantes supera el 5 %.
