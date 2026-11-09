# Análisis estadístico: decisiones y criterios

## Qué queremos saber

Si el corazón se comporta distinto mientras los participantes miran videos de arañas, comparado con un momento de reposo, y si esa respuesta cambia a medida que ven más videos. No buscamos diagnosticar ansiedad con el ECG, solo ver si hay diferencias en la actividad cardíaca entre condiciones.

## De dónde salen los datos que analizo

**Dataset:** ECG de personas con miedo a las arañas mientras miran 16 videos de arañas de 1 minuto cada uno, seguidos de 5 minutos de reposo (Ihmig et al., 2020; PhysioNet). Registrado con BITalino a 100 Hz.

**Pipeline**

```
ECG crudo → filtrado → detección de QRS → serie RR → serie NN → ventanas → métricas HR/HRV → un valor por persona → tests
```

1. **Carga.** Se lee el ECG (en mV) y el eje de tiempo se arma como número de muestra / 100 Hz. Los triggers dicen cuándo empezó y terminó cada video y el reposo. El orden en que cada persona vio los videos se saca de esos horarios, porque fue al azar.
2. **Filtrado.** Pasabanda Butterworth de 0,5 a 40 Hz, aplicado ida y vuelta para no desfasar la señal. El pasa-altos saca la deriva de la línea de base; el pasa-bajos, el ruido de alta frecuencia.
3. **Detección de QRS (Pan-Tompkins).** Pasabanda de 5 a 15 Hz, derivada, cuadrado, promedio móvil de 150 ms y umbrales que se adaptan a la señal. Da la posición de cada pico R.
4. **Serie RR.** El tiempo entre picos R consecutivos.
5. **Serie NN (limpieza).** Se descartan los intervalos que no pueden ser latidos reales: fuera de 300–2000 ms (30 a 200 lpm; Task Force, 1996), o que cambian más de un 20 % respecto del anterior. Los descartados no se rellenan: interpolar inventa valores y baja artificialmente la HRV.
6. **Ventanas.** La serie se corta según los eventos: una ventana de 60 s por video (más seis de 10 s) y ventanas de 60 s en el reposo. Una ventana es válida si conserva al menos el 70 % de los latidos esperados.
7. **Métricas.** En cada ventana se calculan las métricas de HR y HRV (sección siguiente).
8. **Un valor por persona.** Se promedian las ventanas válidas de cada persona en cada condición (expuesto vs reposo)
9. **Tests.** Se comparan las condiciones con tests estadísticos para obtener conclusiones.

Los pasos 1 a 5 son el procesamiento de la señal, el 6 y el 7 la extracción de características, y el 8 y el 9 el análisis estadístico.

## Métricas

- **HR (frecuencia cardíaca):** qué tan rápido late el corazón.
- **HRV (variabilidad de la frecuencia cardíaca):** qué tan irregular es el tiempo entre latidos.

Con estrés, en general se activa el sistema nervioso autónomo simpático, la HR sube y la HRV baja.

| Métrica | Tipo | Qué indica | Fuente |
|---|---|---|---|
| Nmean | HR | HR media comparada con el reposo de la misma persona (lpm por encima o por debajo) | Ihmig et al. 2020, ec. 1 |
| avNN | HR | Tiempo medio entre latidos (ms) | Ihmig et al. 2020, ec. 7 |
| std | HRV | Cuánto se dispersa la HR latido a latido (lpm) | Ihmig et al. 2020, ec. 3 |
| NFD | HRV | Cuánto cambia la HR de un latido al siguiente | Ihmig et al. 2020, ec. 4 |
| NSD | HRV | Lo mismo, pero a dos latidos de distancia | Ihmig et al. 2020, ec. 5 |
| HRV (Ihmig) | Tendencia | Si la HR sube o baja dentro de la ventana. Pese al nombre, no mide variabilidad: la fórmula se reduce a (último intervalo − primero) / N | Ihmig et al. 2020, ec. 6 |
| SDNN | HRV | Variabilidad total de los intervalos (ms) | Ihmig et al. 2020, ec. 8; Task Force 1996 |
| rMSSD | HRV | Variabilidad de corto plazo, entre latidos vecinos. Se asocia al sistema parasimpático ("freno" del corazón) | Ihmig et al. 2020, ec. 9; Task Force 1996 |
| NN50 / pNN50 | HRV | Cantidad / porcentaje de latidos vecinos que difieren en más de 50 ms | Ihmig et al. 2020, ec. 10–11 |
| pNN20 | HRV | Igual, con umbral de 20 ms: capta cambios más chicos | Ihmig et al. 2020, ec. 12 |
| CVNN | HRV | SDNN dividido el intervalo medio: variabilidad relativa a la HR de cada uno | Shaffer & Ginsberg 2017 |
| SD1 / SD2 | HRV | Ancho y largo de la nube del gráfico de Poincaré (corto y largo plazo) | Shaffer & Ginsberg 2017; Rahman et al. 2018 |

Además hay tres medidas que comparan dos momentos del experimento. Por eso dan un valor por persona y no por ventana:

| Medida | Qué indica | Fuente |
|---|---|---|
| ΔHR | HR en los videos − HR en el reposo | Castaneda & Segerstrom 2004 |
| Respuesta inicial | HR en los primeros 10 s del video − HR en los 10 s previos. Positivo es aceleración (defensa); negativo, desaceleración (atención) | Hare & Blevings 1975 |
| Habituación | Si la HR o la HRV de cada persona sube o baja a medida que ve más videos (ver abajo) | Matthews et al. 2015 |

**Cómo se mide la habituación.** Se hace por separado para cada persona y cada métrica:

1. **Se ordenan sus 16 videos según cuántos llevaba vistos,** no según el número del video. Por ejemplo, VP33 vio primero el CLIP-4, después el CLIP-1, después el CLIP-3. Lo que se busca es el efecto de la **posición** (¿influye que sea su 1º, 2º o 16º video?), no el de **cuál** video era.
2. **Se grafica la métrica en cada uno de esos 16 videos** y se traza la recta que mejor pasa por los puntos.
3. **La pendiente de esa recta** es cuánto cambia la métrica, en promedio, por cada video que pasa. Multiplicada por los videos, da cuánto cambió del primero al último.

```
HR de VP33 (lpm sobre su reposo)
 +3 │           •
 +2 │     •  •
 +1 │           •
  0 │ • •  •        •                       •
 -1 │                    •
 -2 │                       • •
 -3 │                              •
 -4 │                           •
 -5 │                                 •
    └──────────────────────────────────────── video nº 1 … 16 (en el orden en que los vio)
      pendiente = −0,32 lpm por video  →  unos −4,8 lpm del primero al último
```

Cómo se lee:
- **Pendiente negativa en la HR:** la HR baja video a video.
- **Pendiente positiva en la HRV:** la variabilidad sube video a video.
- **Cerca de 0:** no cambia.

Las dos primeras son "menos activación" a medida que avanza la sesión. Así queda un número por persona, y P3a testea si esos números tienden todos hacia el mismo lado.

**¿Influye la posición o el video en sí?** Para separarlas, el orden de los videos tendría que ser totalmente al azar. No lo fue: se mostraron en **dos mitades fijas**. Las posiciones 1 a 8 siempre fueron los clips 1 a 8, y las posiciones 9 a 16 siempre los clips 9 a 16; solo se mezclaron dentro de cada mitad. Entonces:
- **Dentro de cada mitad,** la posición y el video están separados: si la HR cambia con la posición, no es por un video en particular.
- **Entre las dos mitades,** no se pueden separar: una diferencia entre la primera y la segunda mitad puede ser por el paso del tiempo o porque los clips 9 a 16 generan menos respuesta.



## Con qué métricas nos quedamos y por qué

Las 14 métricas no miden 14 cosas distintas. Por ejemplo, SD1 y rMSSD son casi la misma fórmula, y SD2 sigue casi exactamente a SDNN. Si testeáramos las 14, contaríamos el mismo resultado varias veces y tendríamos que corregir por 14 comparaciones, lo que nos haría perder sensibilidad sin ganar información.

**Criterio:** se calculó la correlación entre métricas dentro de cada persona, y las que correlacionan más de 0,8 se agruparon en una familia. De cada familia se eligió una representante:

1. que sea la métrica estándar en la bibliografía (comparable con otros trabajos);
2. que tenga una interpretación fisiológica clara;
3. que sea un promedio o un porcentaje antes que un conteo (los conteos dependen de cuántos latidos entran en la ventana).

| Familia | Métricas | Representante |
|---|---|---|
| HR | Nmean, avNN | **Nmean** (ya está comparada contra el reposo de cada persona) |
| HRV total | SDNN, SD2, CVNN, std | **SDNN** |
| HRV de corto plazo | rMSSD, SD1, pNN50, NN50, NFD, NSD | **rMSSD** |
| HRV, cambios chicos | pNN20 | **pNN20** |
| Tendencia | HRV (Ihmig) | **HRV (Ihmig)** |

**Ventanas de 10 s.** En 10 s entran solo unos 12 latidos, y con tan pocos algunas métricas dan valores que saltan mucho. Para ver cuáles sirven:

- Se cortó cada video de 60 s en seis tramos de 10 s. Dentro de un mismo video la situación es la misma, así que los seis tramos deberían dar valores parecidos. Lo que varíen entre sí es **ruido**.
- Ese ruido se comparó con cuánto cambia la métrica **de un video a otro**, que es lo que queremos detectar.
- Una métrica sirve a 10 s si su ruido es menor que esos cambios. Si no, a 10 s mide más ruido que dato.

Solo cumplen **Nmean, avNN, std y rMSSD**. Las demás se usan solo en ventanas de 60 s.

## Decisiones de diseño

| Decisión | Por qué |
|---|---|
| **Solo Group3** (18 personas) | En Group2 y Group4 el reposo viene justo después de un ejercicio de respiración lenta (biofeedback), que sube la variabilidad a propósito. Como el reposo es nuestra referencia, en esos grupos la comparación quedaría alterada. Group3 no hizo ese ejercicio. La única excepción es la replicación de P3b, que no usa el reposo |
| **Ventanas de 60 s** como análisis principal | Cada video dura unos 60 s, así que una ventana es un video entero. Ihmig et al. usan 10 y 60 s, y Pereira et al. (2017) muestran que la HRV es usable en ventanas de alrededor de 1 minuto |
| **El reposo se corta en ventanas de 60 s** | Para comparar videos y reposo con el mismo largo, porque varias métricas cambian con la duración de la ventana |
| **Ventana válida** si conserva al menos el 70 % de los latidos esperados | Con menos, la ventana quedó muy recortada por los latidos descartados y sus métricas no representan ese minuto. Es un criterio nuestro, sin cita |
| **Un valor por persona** antes de cada test | Las ventanas de una misma persona no son independientes. Testearlas como si lo fueran infla el n y da p-valores falsamente chicos (Hurlbert, 1984) |
| **Comparaciones pareadas** | Cada persona se compara contra sí misma, lo que saca del medio las diferencias de HR entre personas |

**Por qué no hacemos análisis en frecuencia (Lomb, LF/HF).**
- **Las ventanas son muy cortas.** La banda LF arranca en 0,04 Hz, o sea, oscilaciones que tardan 25 s en completar un ciclo. En una ventana de 60 s entran apenas dos, y en una de 10 s ni una. Para estimar LF y HF de forma confiable se recomiendan al menos 5 minutos (Task Force, 1996).
- **No hay 5 minutos seguidos de exposición.** Cada video dura 1 minuto y entre videos hay pausas con preguntas. Juntar 5 minutos seguidos mezclaría videos con pausas, que son condiciones distintas. El único tramo de 5 minutos bajo una misma condición es el reposo, y no hay contra qué compararlo.
- Con señales tan cortas, el análisis tiene que basarse en el dominio temporal.

## Tests estadísticos

| Herramienta | Para qué | Por qué esta |
|---|---|---|
| **Wilcoxon de rangos con signo** (test principal) | Comparar dos condiciones en las mismas personas, o un valor contra 0 | No supone que los datos sean normales. Las métricas de HRV suelen ser asimétricas |
| **t de Student pareada** (sensibilidad) | Repetir la comparación suponiendo normalidad | Para verificar que la conclusión no depende del test. |
| **Hodges-Lehmann** | Cuánto cambió la métrica, con su intervalo de confianza del 95 % | Se calcula con la misma lógica que Wilcoxon, así que los dos siempre coinciden: si el test da significativo (p < 0,05), el intervalo queda entero de un lado del 0 (el cambio va claramente para un lado) |
| **r rango-biserial** | Tamaño del efecto (de −1 a 1) | El p dice si la diferencia es probablemente real; r dice si es grande o chica |
| **Holm** | Corregir por hacer varios tests a la vez | Con 5 tests, la chance de algún falso positivo por azar sube bastante. Holm la vuelve a 5 %  |
| **Friedman** | Comparar las 16 posiciones de video | Es como Wilcoxon pero para muchas condiciones. Dentro de cada persona ordena sus 16 videos de menor a mayor HR (o HRV) y mira si algunas posiciones quedan siempre arriba o siempre abajo. Si p < 0,05, la posición del video importa. No dice hacia dónde: eso lo dicen las pendientes |
| **W de Kendall** | Tamaño de efecto de Friedman (de 0 a 1) | Qué tanto coinciden las personas en ese orden. 0 = cada una ordena distinto; 1 = todas igual. 0,1 chico, 0,3 mediano, 0,5 grande |

**Nivel de significancia:** α = 0,05, fijado de antemano.

**Análisis de sensibilidad.** Repetimos el análisis cambiando el test para ver si los resultados se sostienen:
- con la t en lugar de Wilcoxon;
- sin VP03, que tiene el reposo incompleto (190 de 300 s).

## Preguntas, cómo se testean y cómo se leen

| Pregunta | Datos | Test | Cómo leer el resultado |
|---|---|---|---|
| **P1.** ¿Difieren la HR y la HRV entre videos y reposo? | Por persona: promedio de sus ventanas de 60 s en los videos y en el reposo, para las 5 representantes | Wilcoxon por métrica + Holm | HR más alta y HRV más baja en los videos es compatible con más activación |
| **P2.** ¿La HR cambia apenas arranca el video? | Respuesta inicial promedio de los 16 videos, por persona | Wilcoxon contra 0 | Mayor que 0 es aceleración; menor que 0, desaceleración |
| **P3a.** ¿La respuesta baja a lo largo de los 16 videos? | Pendiente de cada métrica según el orden de los videos, por persona | Wilcoxon de las pendientes contra 0 + Holm; Friedman | Si la HR baja o la HRV sube video a video, hay habituación. Sirve también de control de P1: si la variabilidad sube todo el tiempo, la diferencia con el reposo (que es lo último) podría deberse al paso del tiempo y no a las arañas |
| **P3b.** ¿La respuesta baja dentro de cada video? | Primera contra última ventana de 10 s de cada video, promediado por persona (métricas estables a 10 s) | Wilcoxon pareado + Holm | HR más baja al final del video indica habituación dentro del estímulo |

### Dónde está cada resultado

Todo se genera con `python modulos/estadisticas.py` y queda en `results/tables/`:

| Archivo | Pregunta |
|---|---|
| `p1_exposicion_vs_reposo.csv` | P1 |
| `p2_respuesta_inicial.csv` | P2 |
| `p3a_pendientes.csv` | P3a, test de las pendientes |
| `p3a_friedman.csv` | P3a, Friedman |
| `p3b_dentro_del_video.csv` | P3b |
| `p3b_replicacion.csv` | P3b repetido en Group2 y Group4 |

### Cómo leer las tablas

| Columna | Qué mirar |
|---|---|
| `diferencia` | Cuánto cambió, en las unidades de la métrica. En P1 es videos − reposo; en P2, después − antes del inicio; en P3a, cambio por cada video; en P3b, últimos − primeros 10 s |
| `ic95_inf`, `ic95_sup` | Entre qué valores está probablemente el cambio. Si el intervalo cruza el 0, no se puede afirmar que haya cambio |
| `efecto` (r) | Alrededor de 0,1 es chico, 0,3 mediano, 0,5 o más grande. El signo da la dirección |
| `w_kendall` | Efecto de Friedman: 0,1 chico, 0,3 mediano, 0,5 grande |
| `p` | p-valor sin corregir |
| `p_holm` | p corregido. **La conclusión sale de este** |
| `significativo` | Verdadero si `p_holm` < 0,05 |
| `sentido` | Traduce el signo a "más" o "menos activación" (en la HR, bajar es calmarse; en la HRV, subir). Leerlo solo si es significativo |
| `coinciden` | Si Wilcoxon y la t llegan a la misma conclusión |
| columnas con `t_` | Lo mismo, calculado con la t (sensibilidad) |

## Resultados

**P1** (Group3, n = 18, ventanas de 60 s):

| Métrica | Cambio videos − reposo | IC 95 % | r | p | p Holm |
|---|---|---|---|---|---|
| Nmean | +2,0 lpm | −0,3 a 3,9 | 0,44 | 0,11 | 0,33 |
| SDNN | −9,3 ms | −20,3 a −0,7 | −0,59 | 0,027 | 0,13 |
| rMSSD | −5,0 ms | −10,3 a 0,2 | −0,51 | 0,060 | 0,24 |
| pNN20 | −4,1 % | −9,5 a 3,1 | −0,36 | 0,20 | 0,39 |
| HRV (Ihmig) | +0,19 | −0,1 a 0,8 | 0,32 | 0,25 | 0,39 |

**Qué dice P1**

- **La dirección es la esperada.** Mientras miran los videos, la HR es un poco más alta y la HRV un poco más baja que en el reposo. Pasa en 2 de cada 3 personas.
- **Pero no alcanza para afirmarlo.** Después de corregir por hacer 5 tests (Holm), ninguna métrica queda significativa. Tomando los 16 videos juntos, no podemos decir que haya diferencia con el reposo.
- **¿Por qué no da, si la tendencia está? Por los promedios.** P1 promedia los 16 videos de cada persona. Al principio de la sesión la HR está claramente por encima del reposo; al final, ya no:

  | | HR comparada con el reposo |
  |---|---|
  | Primeros 4 videos | +4,1 lpm |
  | Últimos 4 videos | +0,2 lpm |
  | Los 16 videos juntos (P1) | +2,0 lpm |

  Al mezclar los videos del principio, con mucha activación, con los del final, ya tranquilos, el efecto queda a mitad de camino y se diluye. Además, con 18 personas es difícil detectar diferencias moderadas.
- **No depende del método:** con la t se llega a lo mismo, y también sacando a VP03.

**Ojo al leer la tabla:** lo que importa es el cambio de cada persona, no las medianas de cada condición por separado. En rMSSD, la mediana en los videos (30,3 ms) y en el reposo (30,5 ms) son casi iguales. Pero si se compara a cada persona contra sí misma, en los videos baja unos 5 ms. Lo que tapa ese cambio es que unas personas tienen mucha HRV y otras poca.

**P2. Respuesta inicial** (n = 18):

| Mediana | Aceleran / frenan | Cambio (HL) | IC 95 % | r | p (Wilcoxon) | p (t) |
|---|---|---|---|---|---|---|
| −0,07 lpm | 8 / 9 | −0,17 lpm | −1,1 a 0,6 | −0,16 | 0,57 | 0,48 |

No hay una respuesta consistente al empezar el video: la mitad de las personas acelera y la otra mitad frena, y el cambio típico es menor a 1 lpm. La t da lo mismo. Tampoco cambia de forma clara entre los primeros y los últimos videos.

Una posible razón: los triggers tienen resolución de 1 s y el eje temporal del ECG puede correrse hasta unos 3 s, así que los tramos de 10 s antes y después del inicio pueden quedar algo corridos. Una respuesta muy rápida, de pocos segundos, podría no verse.

**P3a. Habituación a lo largo de los 16 videos.** Hay un cambio muy marcado.

Pendientes (Wilcoxon contra 0, con Holm sobre 5 métricas):

| Métrica | Cambio del video 1 al 16 (mediana) | Personas | r | p Holm |
|---|---|---|---|---|
| Nmean | −5,1 lpm | 16 de 18 bajan | −0,80 | **0,006** |
| rMSSD | +4,8 ms | 15 de 18 suben | 0,79 | **0,006** |
| pNN20 | +8,2 % | 16 de 18 suben | 0,81 | **0,006** |
| SDNN | +7,7 ms | 15 de 18 suben | 0,51 | 0,12 |
| HRV (Ihmig) | +0,08 | 11 de 18 suben | 0,27 | 0,32 |

Friedman confirma lo mismo: en Nmean, rMSSD y pNN20 la posición del video importa (p Holm < 0,001). El W de Kendall (0,16 a 0,27, de chico a mediano) indica que el patrón es compartido entre personas, aunque con bastante variación individual. SDNN y HRV (Ihmig) no llegan.

La HR baja y la HRV sube a lo largo de la sesión: la activación es alta al principio y se va apagando. La t llega a la misma conclusión en todas.

**Ojo: el cambio se concentra entre las dos mitades** (exploratorio). Mirando cada mitad por separado, que es donde posición y video están separados, la tendencia no es significativa. En cambio, al pasar de la mitad 1 a la 2 hay un salto claro:

| Métrica | Pendiente dentro de la mitad 1 | Pendiente dentro de la mitad 2 | Salto entre mitades (videos 9–10 vs 7–8) |
|---|---|---|---|
| Nmean | baja (12 de 18), p = 0,18 | sin cambio, p = 1,0 | **−2,6 lpm, p = 0,001** |
| rMSSD | sin cambio, p = 0,97 | sube (11 de 18), p = 0,18 | +1,7 ms, p = 0,13 |
| pNN20 | p = 0,52 | p = 0,47 | +2,3 %, p = 0,30 |

Como la mitad 2 siempre fueron los clips 9 a 16, ese salto puede deberse a tres cosas que no podemos separar:
- que la gente se calma con el tiempo;
- que esos clips generan menos respuesta;
- que algo cambió entre las mitades. Según nuestra guía, a partir del clip 9 cambian las preguntas que se hacen entre videos (a confirmar en Ihmig et al., 2020).

Por eso, P3a muestra que la activación es menor en la segunda mitad de la sesión, pero no alcanza para hablar de una habituación gradual video a video.

**Esto cambia cómo se lee P1.** El reposo está al final del experimento. Comparado contra los primeros videos, la diferencia es clara; contra los últimos, desaparece:

| Métrica | Videos 1–4 vs reposo | Videos 13–16 vs reposo |
|---|---|---|
| Nmean | +4,1 lpm (r = 0,71, p = 0,007) | +0,2 lpm (p = 0,83) |
| SDNN | −14,7 ms (r = −0,61, p = 0,02) | −4,9 ms (p = 0,20) |
| rMSSD | −7,0 ms (r = −0,57, p = 0,03) | −0,9 ms (p = 0,67) |

Esta comparación es exploratoria.

**P3b. Habituación dentro de cada video:** últimos 10 s contra primeros 10 s, promediado sobre los 16 videos. Wilcoxon pareado, con Holm sobre 3 métricas estables a 10 s.

| Métrica | Primeros 10 s | Últimos 10 s | Cambio (HL) | IC 95 % | Personas | r | p Holm |
|---|---|---|---|---|---|---|---|
| Nmean | +4,4 lpm | +0,7 lpm | −2,0 lpm | −3,6 a −1,2 | 18 de 18 bajan | −1,00 | **< 0,001** |
| std | 4,9 lpm | 4,0 lpm | −1,0 lpm | −1,3 a −0,8 | 17 de 18 bajan | −0,81 | **0,003** |
| rMSSD | 28,8 ms | 29,0 ms | −0,1 ms | −0,9 a 1,7 | 11 de 18 bajan | −0,04 | 0,90 |

La t llega a la misma conclusión en las tres.

- **La HR baja dentro de cada video, en las 18 personas.** Mirando las 6 ventanas de 10 s, la HR arranca unos 4 lpm por encima del reposo y en unos 20 s ya está en el nivel del reposo, donde se queda hasta el final del video.
- **std baja porque al principio la HR se está moviendo,** justamente al bajar, no porque cambie la activación. Por eso no la interpretamos como "más" o "menos" activación.
- **rMSSD no cambia** dentro del video.

**P3b se repite en los otros dos grupos** (exploratorio). P3b compara cada video consigo mismo, así que no necesita el reposo y se puede hacer también en Group2 y Group4. Esos grupos tienen biofeedback durante toda la sesión, por eso solo los usamos para esta comparación y no comparamos sus niveles con los de Group3.

| Grupo | Personas | Cambio de HR (HL) | IC 95 % | r | p Holm |
|---|---|---|---|---|---|
| Group3 | 18 de 18 bajan | −2,0 lpm | −3,6 a −1,2 | −1,00 | **< 0,001** |
| Group2 | 18 de 19 bajan | −2,1 lpm | −2,5 a −1,7 | −0,91 | **< 0,001** |
| Group4 | 14 de 19 bajan | −1,6 lpm | −2,9 a −0,4 | −0,64 | **0,037** |

La baja de la HR dentro del video aparece en los tres grupos, con un tamaño parecido (unos 2 lpm), y la t llega a lo mismo. En total son 56 personas. La figura `12_replicacion_por_grupo.png` muestra que en los tres la HR baja en los primeros 20 s y después se mantiene. En cambio, la baja de std no se repite en Group2 ni en Group4. rMSSD tampoco cambia de forma significativa con Wilcoxon. En Group2 la t sí marca una suba de rMSSD, pero como los dos tests no coinciden, no la tomamos como resultado. Lo único que se replica es la HR.

**Un dato que cambia la lectura** (exploratorio): en los 10 s *antes* de cada video, la HR ya está unos 4 a 6 lpm por encima del reposo (media +4,1; mediana +5,6), en 13 de 18 personas. Es decir, la HR no sube cuando aparece la araña (por eso P2 no da nada): ya estaba alta antes. Puede ser anticipación del próximo video, o efecto de lo que hacían entre videos (responder preguntas sobre el video anterior). Con este diseño no podemos separar las dos cosas.

## Conclusiones

**Lo que muestran los datos**

1. **Durante la sesión de videos hay más activación cardíaca que en el reposo, sobre todo al principio.** En los primeros videos la HR está unos 4 lpm por encima del reposo y la HRV (SDNN, rMSSD) es claramente menor (comparación exploratoria).
2. **Esa activación es menor en la segunda mitad de la sesión (P3a).** Del primer al último video la HR baja unos 5 lpm y la HRV sube, en 15 o 16 de las 18 personas. Es significativo después de Holm. Hacia el final ya no se distingue del reposo. Pero el cambio se concentra en el paso de la mitad 1 a la mitad 2, más que en una baja gradual video a video.
3. **Y también se apaga dentro de cada video (P3b).** La HR arranca alta y en unos 20 s baja unos 2 lpm; en Group3 eso la deja en el nivel del reposo. Pasa en las 18 personas de Group3 y se repite en Group2 y Group4: en total, en 50 de 56 personas. Es el resultado más sólido del trabajo.
4. **La HR no sube cuando aparece la araña (P2): ya estaba alta antes.** En los 10 s previos a cada video la HR ya está unos 4 a 6 lpm por encima del reposo (exploratorio). La activación parece empezar antes del video, no con él.
5. **Por eso, tomando los 16 videos completos, no hay diferencia significativa con el reposo (P1).** La dirección es la esperada y los efectos son medianos a grandes, pero al promediar los momentos activados con los ya tranquilos el efecto se diluye. Con 18 personas, además, falta poder para detectarlo.

**En una frase:** la activación cardíaca es transitoria. Es máxima antes y al empezar cada video, se apaga en unos segundos (esto se repite en los tres grupos), y es menor en la segunda mitad de la sesión que en la primera.

**Qué tan firmes son estos resultados**

- Los resultados no dependen del test elegido: Wilcoxon y la t llegan a la misma conclusión en todas las comparaciones de Group3. La única diferencia aparece en la replicación, en rMSSD de Group2, y no afecta a ninguna conclusión.
- No dependen de VP03: sacarlo, debido a que su grabación se corto antes de tiempo y el reposo, que usamos como referencia, no es confiable; no cambia nada. Por eso hicimos un análisis de sensibilidad: repetir P1 sin VP03 no cambió nada. Por lo tanto, afirmamos que los resultados no dependen de incluir o no al participante con el reposo incompleto.
- La baja de la HR dentro del video (P3b) no es propia de Group3: aparece también en Group2 y Group4, que son personas distintas y con otra condición experimental (biofeedback durante los videos). Que el mismo análisis dé lo mismo en tres grupos independientes hace que no sea un resultado casual de una sola muestra.


**Qué no podemos afirmar**

- **No sabemos por qué la activación es menor en la segunda mitad.** Puede ser que se acostumbran a las arañas, que se relajan con el tiempo, que los clips 9 a 16 generan menos respuesta, o el cambio de preguntas a partir del clip 9. Los videos se mostraron en dos mitades fijas, así que el diseño no permite separar estas causas.
- **No sabemos si la activación antes de cada video es miedo anticipado o efecto de responder las preguntas** sobre el video anterior.
- **Los resultados valen para Group3** (18 personas sin entrenamiento de biofeedback). La excepción es la baja de la HR dentro del video, que también comprobamos en Group2 y Group4. El resto (P1, P2, P3a) no se puede extender a esos grupos, porque usa el reposo o compara momentos de la sesión en los que estaban haciendo biofeedback.
- **El ECG no diagnostica ansiedad.** Lo que mostramos son cambios en la actividad cardíaca entre condiciones, compatibles con más activación durante la exposición.


## Referencias

- Castaneda, J. O., & Segerstrom, S. C. (2004). *J Anxiety Disord*, 18(6), 809–823.
- Friedman, M. (1937). *J Am Stat Assoc*, 32(200), 675–701.
- Hare, R. D., & Blevings, G. (1975). *Biol Psychol*, 3(1), 1–13.
- Hodges, J. L., & Lehmann, E. L. (1963). *Ann Math Stat*, 34(2), 598–611.
- Holm, S. (1979). *Scand J Stat*, 6(2), 65–70.
- Hurlbert, S. H. (1984). *Ecol Monogr*, 54(2), 187–211.
- Ihmig, F. R., et al. (2020). *PLoS ONE*, 15(6), e0231517.
- Kendall, M. G., & Babington Smith, B. (1939). *Ann Math Stat*, 10(3), 275–287.
- Kerby, D. S. (2014). *Comprehensive Psychology*, 3, 11.IT.3.1.
- Lipponen, J. A., & Tarvainen, M. P. (2019). *J Med Eng Technol*, 43(3), 173–181.
- Matthews, A., Naran, N., & Kirkby, K. C. (2015). *J Behav Ther Exp Psychiatry*, 47, 129–137.
- Pereira, T., et al. (2017). *Comput Methods Programs Biomed*.
- Rahman, S., Habel, M., & Contrada, R. J. (2018). *Int J Psychophysiol*.
- Rochon, J., & Kieser, M. (2011). *Br J Math Stat Psychol*, 64(3), 410–426.
- Rochon, J., Gondan, M., & Kieser, M. (2012). *BMC Med Res Methodol*, 12, 81.
- Shaffer, F., & Ginsberg, J. P. (2017). *Front Public Health*, 5, 258.
- Student (1908). *Biometrika*, 6(1), 1–25.
- Task Force of the ESC and NASPE (1996). *Circulation*, 93(5), 1043–1065.
- Wilcoxon, F. (1945). *Biometrics Bulletin*, 1(6), 80–83.
