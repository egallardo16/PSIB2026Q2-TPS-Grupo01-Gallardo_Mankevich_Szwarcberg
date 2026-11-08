# Análisis estadístico: decisiones y criterios

## Qué queremos saber

Si el corazón se comporta distinto mientras los participantes miran videos de arañas, comparado con un momento de reposo, y si esa respuesta cambia a medida que ven más videos. No buscamos diagnosticar ansiedad con el ECG, solo ver si hay diferencias en la actividad cardíaca entre condiciones.

## De dónde salen los datos que analizo

1. **Señal → latidos.** Se filtra el ECG (0,5 a 40 Hz), se detecta cada latido con Pan-Tompkins y se arma la serie de intervalos entre latidos. Se descartan los intervalos que no pueden ser latidos reales: fuera de 300–2000 ms, o que cambian más de un 20 % respecto del anterior.
2. **Latidos → métricas.** La serie se corta en ventanas y en cada una se calculan las métricas.
3. **Métricas → conclusiones.** Es esta parte: se resume a un valor por persona y se aplican los tests.

## Métricas

**Nivel** dice qué tan rápido late el corazón. **Variabilidad** dice qué tan irregular es el tiempo entre latidos. Con estrés, en general el nivel sube y la variabilidad baja.

| Métrica | Tipo | Qué indica | Fuente |
|---|---|---|---|
| Nmean | Nivel | Pulso medio comparado con el reposo de la misma persona (lpm por encima o por debajo) | Ihmig et al. 2020, ec. 1 |
| avNN | Nivel | Tiempo medio entre latidos (ms) | Ihmig et al. 2020, ec. 7 |
| std | Variabilidad | Cuánto se dispersa el pulso latido a latido (lpm) | Ihmig et al. 2020, ec. 3 |
| NFD | Variabilidad | Cuánto cambia el pulso de un latido al siguiente | Ihmig et al. 2020, ec. 4 |
| NSD | Variabilidad | Lo mismo, pero a dos latidos de distancia | Ihmig et al. 2020, ec. 5 |
| HRV (Ihmig) | Tendencia | Si el pulso sube o baja dentro de la ventana. Pese al nombre, no mide variabilidad: la fórmula se reduce a (último intervalo − primero) / N | Ihmig et al. 2020, ec. 6 |
| SDNN | Variabilidad | Variabilidad total de los intervalos (ms) | Ihmig et al. 2020, ec. 8; Task Force 1996 |
| rMSSD | Variabilidad | Variabilidad de corto plazo, entre latidos vecinos. Se asocia al sistema parasimpático ("freno" del corazón) | Ihmig et al. 2020, ec. 9; Task Force 1996 |
| NN50 / pNN50 | Variabilidad | Cantidad / porcentaje de latidos vecinos que difieren en más de 50 ms | Ihmig et al. 2020, ec. 10–11 |
| pNN20 | Variabilidad | Igual, con umbral de 20 ms: capta cambios más chicos | Ihmig et al. 2020, ec. 12 |
| CVNN | Variabilidad | SDNN dividido el intervalo medio: variabilidad relativa al pulso de cada uno | Shaffer & Ginsberg 2017 |
| SD1 / SD2 | Variabilidad | Ancho y largo de la nube del gráfico de Poincaré (corto y largo plazo) | Shaffer & Ginsberg 2017; Rahman et al. 2018 |

Además hay tres medidas que comparan dos momentos del experimento. Por eso dan un valor por persona y no por ventana:

| Medida | Qué indica | Fuente |
|---|---|---|
| ΔHR | Pulso en los videos − pulso en el reposo | Castaneda & Segerstrom 2004 |
| Respuesta inicial | Pulso en los primeros 10 s del video − pulso en los 10 s previos. Positivo es aceleración (defensa); negativo, desaceleración (atención) | Hare & Blevings 1975 |
| Habituación | Pendiente de cada métrica a lo largo de los 16 videos, en el orden en que cada persona los vio | Matthews et al. 2015 |

La profesora pidió sumar métricas de otros papers además de las de Ihmig: de ahí salen CVNN, SD1/SD2 y las tres medidas de respuesta.

## Con qué métricas nos quedamos y por qué

Las 14 métricas no miden 14 cosas distintas. Por ejemplo, SD1 y rMSSD son casi la misma fórmula, y SD2 sigue casi exactamente a SDNN. Si testeáramos las 14, contaríamos el mismo resultado varias veces y tendríamos que corregir por 14 comparaciones, lo que nos haría perder sensibilidad sin ganar información.

**Criterio:** se calculó la correlación entre métricas dentro de cada persona, y las que correlacionan más de 0,8 se agruparon en una familia. De cada familia se eligió una representante con reglas fijadas antes de ver los resultados:

1. que sea la métrica estándar en la bibliografía (comparable con otros trabajos);
2. que tenga una interpretación fisiológica clara;
3. que sea un promedio o un porcentaje antes que un conteo (los conteos dependen de cuántos latidos entran en la ventana).

| Familia | Métricas | Representante |
|---|---|---|
| Nivel de pulso | Nmean, avNN | **Nmean** (ya está comparada contra el reposo de cada persona) |
| Variabilidad total | SDNN, SD2, CVNN, std | **SDNN** |
| Variabilidad de corto plazo | rMSSD, SD1, pNN50, NN50, NFD, NSD | **rMSSD** |
| Cambios chicos | pNN20 | **pNN20** |
| Tendencia | HRV (Ihmig) | **HRV (Ihmig)** |

**Ventanas de 10 s.** En 10 s entran unos 12 latidos, y casi ninguna métrica se puede estimar bien con tan pocos. Se comparó cuánto varía cada métrica entre los seis tramos de 10 s de un mismo video (donde la condición no cambia, así que esa variación es error) contra cuánto varía entre videos distintos. Solo cuatro aguantan: **Nmean, avNN, std y rMSSD**. Las demás se usan solo en ventanas de 60 s.

## Decisiones de diseño

| Decisión | Por qué |
|---|---|
| **Solo Group3** (18 personas) | En Group2 y Group4 el reposo viene justo después de un ejercicio de respiración lenta (biofeedback), que sube la variabilidad a propósito. Como el reposo es nuestra referencia, en esos grupos la comparación quedaría alterada. Group3 no hizo ese ejercicio |
| **Ventanas de 60 s** como análisis principal | Cada video dura unos 60 s, así que una ventana es un video entero. Ihmig et al. usan 10 y 60 s, y Pereira et al. (2017) muestran que la HRV es usable en ventanas de alrededor de 1 minuto |
| **El reposo se corta en ventanas de 60 s** | Para comparar videos y reposo con el mismo largo, porque varias métricas cambian con la duración de la ventana |
| **Ventana válida** si conserva al menos el 70 % de los latidos esperados | Con menos, la ventana quedó muy recortada por los latidos descartados y sus métricas no representan ese minuto. Es un criterio nuestro, sin cita |
| **Un valor por persona** antes de cada test | Las ventanas de una misma persona no son independientes. Testearlas como si lo fueran infla el n y da p-valores falsamente chicos (Hurlbert, 1984) |
| **Comparaciones pareadas** | Cada persona se compara contra sí misma, lo que saca del medio las diferencias de pulso entre personas |
| **Sin Machine Learning** | Lo pidió la profesora: en el TP se hace caracterización estadística |
| **Sin análisis en frecuencia (Lomb)** | La profesora señaló que las señales son demasiado cortas para análisis frecuencial. Ojo: la consigna lista Lomb entre los temas obligatorios, así que hay que justificar esta decisión en el informe |

## Tests estadísticos

| Herramienta | Para qué | Por qué esta |
|---|---|---|
| **Wilcoxon de rangos con signo** (test principal) | Comparar dos condiciones en las mismas personas, o un valor contra 0 | No supone que los datos sean normales. Las métricas de HRV suelen ser asimétricas |
| **t de Student pareada** (sensibilidad) | Repetir la comparación suponiendo normalidad | Para mostrar que la conclusión no depende del test. Nunca decide |
| **Hodges-Lehmann** | Cuánto cambió la métrica, con su intervalo de confianza del 95 % | Es la medida que va con Wilcoxon: si p < 0,05, el intervalo no incluye el 0 |
| **r rango-biserial** | Tamaño del efecto (de −1 a 1) | El p dice si la diferencia es probablemente real; r dice si es grande o chica |
| **Holm** | Corregir por hacer varios tests a la vez | Con 5 tests, la chance de algún falso positivo por azar sube de 5 % a cerca de 23 %. Holm la vuelve a 5 % y es menos estricto que Bonferroni |
| **Friedman** | Comparar las 16 posiciones de video | Es la versión de Wilcoxon para más de dos condiciones en las mismas personas |

**Nivel de significancia:** α = 0,05, fijado de antemano.

**Por qué no elegimos el test según la normalidad.** Una opción común es aplicar primero Shapiro-Wilk y usar la t si da normal o Wilcoxon si no. Ese procedimiento en dos pasos distorsiona la tasa de error del test final (Rochon & Kieser, 2011; Rochon et al., 2012). Además, con 18 personas Shapiro casi nunca detecta que los datos no son normales. Por eso fijamos Wilcoxon de antemano para todo.

**Análisis de sensibilidad.** Repetimos el análisis cambiando una decisión dudosa para ver si la conclusión se sostiene:
- con la t en lugar de Wilcoxon;
- sin VP03, que tiene el reposo incompleto (190 de 300 s).

## Preguntas, cómo se testean y cómo se leen

| Pregunta | Datos | Test | Cómo leer el resultado |
|---|---|---|---|
| **P1.** ¿Difieren pulso y variabilidad entre videos y reposo? | Por persona: promedio de sus ventanas de 60 s en los videos y en el reposo, para las 5 representantes | Wilcoxon por métrica + Holm | Pulso más alto y variabilidad más baja en los videos es compatible con más activación |
| **P2.** ¿El pulso cambia apenas arranca el video? | Respuesta inicial promedio de los 16 videos, por persona | Wilcoxon contra 0 | Mayor que 0 es aceleración; menor que 0, desaceleración |
| **P3a.** ¿La respuesta baja a lo largo de los 16 videos? | Pendiente de cada métrica según el orden de los videos, por persona | Wilcoxon de las pendientes contra 0 + Holm; Friedman | Si el pulso baja o la variabilidad sube video a video, hay habituación. Sirve también de control de P1: si la variabilidad sube todo el tiempo, la diferencia con el reposo (que es lo último) podría deberse al paso del tiempo y no a las arañas |
| **P3b.** ¿La respuesta baja dentro de cada video? | Primera contra última ventana de 10 s de cada video, promediado por persona (métricas estables a 10 s) | Wilcoxon pareado | Pulso más bajo al final del video indica habituación dentro del estímulo |

### Cómo leer la tabla de resultados

| Columna | Qué mirar |
|---|---|
| `diferencia` | Cuánto cambió (videos − reposo), en las unidades de la métrica |
| `ic95_inf`, `ic95_sup` | Entre qué valores está probablemente el cambio. Si el intervalo cruza el 0, no se puede afirmar que haya cambio |
| `efecto` (r) | Alrededor de 0,1 es chico, 0,3 mediano, 0,5 o más grande. El signo da la dirección |
| `p` | p-valor sin corregir |
| `p_holm` | p corregido. **La conclusión sale de este** |
| `significativo` | Verdadero si `p_holm` < 0,05 |
| `coinciden` | Si Wilcoxon y la t llegan a la misma conclusión |

## Resultados hasta ahora

**P1** (Group3, n = 18, ventanas de 60 s):

| Métrica | Cambio videos − reposo | IC 95 % | r | p | p Holm |
|---|---|---|---|---|---|
| Nmean | +2,0 lpm | −0,3 a 3,9 | 0,44 | 0,11 | 0,33 |
| SDNN | −9,3 ms | −20,3 a −0,7 | −0,59 | 0,027 | 0,13 |
| rMSSD | −5,0 ms | −10,3 a 0,2 | −0,51 | 0,060 | 0,24 |
| pNN20 | −4,1 % | −9,5 a 3,1 | −0,36 | 0,20 | 0,39 |
| HRV (Ihmig) | +0,19 | −0,1 a 0,8 | 0,32 | 0,25 | 0,39 |

- En los videos el pulso tiende a ser más alto y la variabilidad más baja, que es la dirección esperada. Pasa en 2 de cada 3 personas: 12 de 18 suben el pulso, y 11 o 12 de 18 bajan SDNN, rMSSD y pNN20.
- SDNN da diferencia antes de corregir, pero ninguna métrica es significativa después de Holm. No se puede afirmar que haya diferencia entre videos y reposo tomando los 16 videos juntos.
- Los efectos son de medianos a grandes (r entre 0,4 y 0,6 en Nmean, SDNN y rMSSD). Que no sean significativos no quiere decir que no haya efecto: con 18 personas falta poder estadístico, y además el efecto se diluye (ver P3a).
- La t llega a la misma conclusión en las 5 métricas, y sacar a VP03 tampoco la cambia. El resultado no depende de esas decisiones.

**Ojo al leer la tabla:** las medianas de cada condición por separado no son el efecto. En rMSSD las medianas de videos y reposo son casi iguales (30,3 y 30,5 ms), pero dentro de cada persona el rMSSD baja unos 5 ms en los videos. Las diferencias entre personas tapan el cambio. Por eso el efecto se lee en `diferencia`, que compara a cada persona contra sí misma.

**P2. Respuesta inicial** (n = 18):

| Mediana | Aceleran / frenan | Cambio (HL) | IC 95 % | r | p (Wilcoxon) | p (t) |
|---|---|---|---|---|---|---|
| −0,07 lpm | 8 / 9 | −0,17 lpm | −1,1 a 0,6 | −0,16 | 0,57 | 0,48 |

No hay una respuesta consistente al empezar el video: la mitad de las personas acelera y la otra mitad frena, y el cambio típico es menor a 1 lpm. La t da lo mismo. Tampoco cambia de forma clara entre los primeros y los últimos videos.

Una posible razón: los triggers tienen resolución de 1 s y el eje temporal del ECG puede correrse hasta unos 3 s, así que los tramos de 10 s antes y después del inicio pueden quedar algo corridos. Una respuesta muy rápida, de pocos segundos, podría no verse.

**P3a** (resultado preliminar, todavía sin Holm): **hay un cambio muy marcado a lo largo de los 16 videos.**

| Métrica | Cambio del video 1 al 16 (mediana) | Personas que siguen esa tendencia | p |
|---|---|---|---|
| Nmean | −5,1 lpm | 16 de 18 bajan | 0,002 |
| rMSSD | +4,8 ms | 15 de 18 suben | 0,002 |
| pNN20 | +8,2 % | 16 de 18 suben | 0,001 |
| SDNN | +7,7 ms | 15 de 18 suben | 0,060 |

El pulso baja y la variabilidad sube video a video: la activación es alta al principio y se va apagando.

**Esto cambia cómo se lee P1.** El reposo está al final del experimento. Comparado contra los primeros videos, la diferencia es clara; contra los últimos, desaparece:

| Métrica | Videos 1–4 vs reposo | Videos 13–16 vs reposo |
|---|---|---|
| Nmean | +4,1 lpm (r = 0,71, p = 0,007) | +0,2 lpm (p = 0,83) |
| SDNN | −14,7 ms (r = −0,61, p = 0,02) | −4,9 ms (p = 0,20) |
| rMSSD | −7,0 ms (r = −0,57, p = 0,03) | −0,9 ms (p = 0,67) |

Esta comparación es exploratoria: la armamos después de ver la habituación, así que no reemplaza a P1 y hay que presentarla como tal.

**P3b:** pendiente.

## Conclusiones

*Se actualizan a medida que se completan P3a y P3b.*

**Lo que muestran los datos**

1. **Los videos de arañas aumentan la activación cardíaca, pero sobre todo al principio.** En los primeros videos el pulso está unos 4 lpm por encima del reposo y la variabilidad (SDNN, rMSSD) es claramente menor. Es la respuesta esperable ante un estímulo que genera miedo.
2. **Esa activación se apaga a lo largo de la sesión.** Del primer al último video el pulso baja unos 5 lpm y la variabilidad sube, en 15 o 16 de las 18 personas. Hacia el final ya no se distingue del reposo.
3. **Por eso, tomando los 16 videos juntos, no hay diferencia significativa con el reposo (P1).** La dirección es la esperada y los efectos son medianos a grandes, pero al promediar los primeros videos (muy activados) con los últimos (ya tranquilos), el efecto se diluye. Con 18 personas, además, falta poder para detectarlo.
4. **No hay una respuesta consistente en los primeros segundos de cada video (P2).** La mitad de las personas acelera y la otra mitad frena. Puede que la respuesta sea muy rápida para verla con triggers de 1 s de resolución.

**Qué tan firmes son estos resultados**

- No dependen del test elegido: Wilcoxon y la t llegan a la misma conclusión en todas las comparaciones.
- No dependen de VP03: sacarlo, por su reposo incompleto, no cambia nada.
- La comparación primeros vs últimos videos contra el reposo es exploratoria, porque la armamos después de ver la habituación. Sirve para explicar P1, no para reemplazarlo.

**Qué no podemos afirmar**

- **No sabemos si la respuesta baja porque se acostumbran a las arañas o porque se relajan con el tiempo.** El reposo está al final y no hay una medición antes de los videos, así que el diseño no permite separar las dos cosas.
- **Los resultados valen para Group3** (18 personas sin entrenamiento de biofeedback). No se pueden extender a los otros grupos sin analizarlos.
- **El ECG no diagnostica ansiedad.** Lo que mostramos son cambios en la actividad cardíaca entre condiciones, compatibles con más activación durante la exposición.

## Cosas abiertas

- **Video de demostración como referencia previa.** Es lo único que se grabó antes de los clips de arañas. Si su contenido es neutro, podría servir como una línea de base anterior a la exposición y ayudar a separar habituación de paso del tiempo. Hay que confirmar en el protocolo qué muestra ese video.
- **HRV (Ihmig) en la familia principal.** Mide tendencia y no variabilidad. Podríamos testear solo 4 métricas en P1 y reportarla aparte como exploratoria.
- **Criterio del 20 % para descartar intervalos.** En algunos participantes descarta latidos normales: cambios del pulso con la respiración, no errores de detección. Eso puede bajar un poco rMSSD y SDNN. Una alternativa es comparar cada intervalo contra la mediana de los latidos cercanos (Lipponen & Tarvainen, 2019).
- **Comparación entre grupos** (Kruskal-Wallis). Con solo Group3 no se puede hacer; habría que calcular las métricas para los 57 participantes.
- **Lomb.** Justificar en el informe por qué no se hace.

## Referencias

- Castaneda, J. O., & Segerstrom, S. C. (2004). *J Anxiety Disord*, 18(6), 809–823.
- Hare, R. D., & Blevings, G. (1975). *Biol Psychol*, 3(1), 1–13.
- Hodges, J. L., & Lehmann, E. L. (1963). *Ann Math Stat*, 34(2), 598–611.
- Holm, S. (1979). *Scand J Stat*, 6(2), 65–70.
- Hurlbert, S. H. (1984). *Ecol Monogr*, 54(2), 187–211.
- Ihmig, F. R., et al. (2020). *PLoS ONE*, 15(6), e0231517.
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
