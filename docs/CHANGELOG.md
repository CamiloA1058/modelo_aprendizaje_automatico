## [Validación] V1: ninguna selección con los datos de prueba - 2026-10-16

Un revisor académico señaló que el procedimiento de validación y entrenamiento "requiere verificación". Principio adoptado: **nada de lo que se reporta como resultado se elige, ajusta o selecciona con los mismos datos sobre los que se reporta** (pesos, hiperparámetros, umbrales, cuantiles, elección de modelo).

- La selección usa solo la partición de entrenamiento: validación cruzada temporal (pliegues mensuales) o una **validación interna temporal** dentro del entrenamiento (los primeros ~80 % de los meses de entrenamiento de cada producto entrenan un modelo temporal; el último ~20 % lo valida; misma regla de `_build_train_frame`, aplicada una segunda vez).
- La **prueba interna** (último 20 % de los meses de cada producto) solo reporta.
- El **estudio de caso** (origen rodante sobre los seis meses del holdout, cada mes entrenado solo con datos anteriores) es la evaluación final fuera de muestra.

### Auditoría de las selecciones

| Elemento | Dónde se elegía | Dónde se reporta | ¿Limpio antes? | Corrección |
|---|---|---|---|---|
| Peso w de la mezcla w·RF + (1−w)·media_3 | Prueba interna del regresor (`train_regressor`) | Prueba interna (`evaluacion_modelo.txt`: R² 0,4799, razón 1,0126) y estudio de caso | **No** | Validación interna: regresor temporal (mismos hiperparámetros, log1p, top productos, escalador y KMeans reajustados solo con el entrenamiento interno) y criterio de sesgo de la suma |
| Hiperparámetros del clasificador | GridSearchCV F1 macro en pliegues mensuales del entrenamiento; luego se **conservó** `max_depth=14` porque la prueba interna daba parecido | Prueba interna y estudio de caso | **No** (la decisión final miraba la prueba) | `DEFAULT_CLF_PARAMS` = ganador de la CV `{300, None, 3}`; `tune_rf.py` lo declara y deja la comparación de prueba solo como información |
| Umbral de ¿se venderá? | Prueba interna del clasificador binario (`te`) | Misma prueba interna (79,3 %, tarjeta del dashboard) y estudio de caso | **No** | Validación interna (clasificador temporal con el entrenamiento interno); la exactitud se reporta en la prueba interna (`sale_test_accuracy`) |
| Cuantiles del stock sugerido | Residuos de la prueba interna del regresor (con el w elegido ahí) y del clasificador de venta | Cobertura solo en el estudio de caso | Sí respecto del holdout; **no** respecto de "la prueba interna solo reporta" | Residuos de la validación interna |
| k de KMeans | Silueta y codo con la partición de entrenamiento (`evaluate_k`) | Silueta como criterio, no como resultado | Sí | Sin cambio |
| Escaladores (RobustScaler) y KMeans de producción | Ajustados solo con la partición de entrenamiento | — | Sí | Sin cambio (la validación interna reajusta los suyos con el entrenamiento interno) |
| Umbrales de clase 1,15 / 0,90 | Valores fijos desde la primera versión del repositorio (2026-05-21); no hay registro de que se ajustaran con datos | — | Sí, hasta donde muestra el historial | Sin cambio |
| Top 1.900 productos | Cuáles: suma de ventas de la partición de entrenamiento. Cuántos (1.900): valor fijo heredado, sin registro de ajuste | — | Sí, hasta donde muestra el historial | Sin cambio |
| XGBoost en `compare_models.py` | GridSearchCV F1 macro en los mismos pliegues mensuales del entrenamiento | Prueba interna | Sí | Sin cambio; el RF usa ahora el ganador de la CV con la misma regla |
| Variantes de `experimento_monto_producto.py` (w, Tweedie, umbral, corrección) | Prueba interna de cada origen | Holdout | Limpio respecto del holdout; usan la prueba interna para elegir | Sin cambio (experimento; la cifra de la tesis es la de producción) |

### Cambios en el código

- `src/ventas_forecast/validacion_interna.py` (nuevo): `particion_temporal(ids, fechas, ratio)`.
- `SalesForecastModel`: `_particion_interna()` (entrenamiento interno / validación interna y KMeans interno), `_validacion_interna_regresor()`, `blend_validacion`, `REG_PARAMS`, `SALE_CLF_PARAMS`, `sale_test_accuracy`; `agregar_probabilidad_venta()` elige el umbral con la validación interna; `_validacion_stock()` usa la validación interna. Sin validación interna posible (datos mínimos), w = 1 (RF puro) y umbral 0,5.
- `DEFAULT_CLF_PARAMS = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3}`.
- Dashboard: tarjeta "Métricas del modelo (prueba interna)"; "Acierto ¿se venderá?" y el KPI muestran `sale_test_accuracy`; el registro dice "Umbral 0,64 (elegido con la validación interna) | exactitud en la prueba interna 79,2 %"; notas de Grupos (69 % / 76 %) y Productos (72 % medido) con las cifras nuevas.
- `scripts/evaluate_model.py` explica cómo se elige w y agrega la sección de probabilidad de venta; `scripts/experimento_probabilidad_venta.py` elige el umbral igual que producción y reporta la exactitud de cada modelo en la prueba interna.
- Pruebas (TDD, RED observado contra el código anterior): `tests/test_seleccion_sin_prueba.py` (12: w y cuantiles no cambian al alterar los objetivos de la prueba interna, el w óptimo en la prueba puede ser otro, el modelo interno no ve los objetivos de la validación, umbral invariable al invertir las etiquetas de prueba, exactitud reportada en la prueba, KMeans interno), `DEFAULT_CLF_PARAMS` fijado al ganador de la CV, notas del dashboard; suite de 408 pruebas.

### Resultado clave: w = 0 en abril

En la validación interna el regresor temporal predice solo el 23,6 % de la suma real (la media de 3 meses, el 94,8 %), así que el w que minimiza el sesgo es **0,0**: la predicción combinada de abril es la media de 3 meses (con el top definido por la partición completa de entrenamiento w también da 0,0). En el estudio de caso w = 0,0 / 0,1 / 0,1 / 0,2 / 0,2 / 0,2 (antes 0,4 / 0,3 / 0,2 / 0,3 / 0,3 / 0,3). El RF puro (`ventas_predichas_rf`) no cambia.

### Antes → después (V6 hasta el 31/03/2026; holdout `data/holdout/ventas_2026_04_09.csv`)

**Clasificador, prueba interna (n = 6.242):** F1 macro 0,4501 → 0,4502; exactitud balanceada 0,5043 → 0,4974; exactitud 0,6684 → 0,6831; F1 ponderado 0,7126 → 0,7202; F1 por clase Reducir 0,7873 → 0,7967, Mantener 0,0816 → 0,0701, Reforzar 0,4814 → 0,4838. Línea base mayoritaria sin cambio (0,2959 / 0,3333 / 0,7980). CV del ganador 0,4377 ± 0,0086 (reconfirmada: tabla de la grilla idéntica).

**Regresor, prueba interna (n = 4.472):** RF, ingenuo y media_3 sin cambio (MAE 232.349 / 427.404 / 314.974; RMSE 907.724 / 1.254.946 / 930.639; R² 0,4213 / −0,1061 / 0,3917; MASE 1,0382 / 1,9097 / 1,4074; razón de sumas RF 0,5937, media_3 1,2919). Mezcla: w 0,4 → 0,0; MAE 266.174 → 314.974; RMSE 860.530 → 930.639; R² 0,4799 → 0,3917; MASE 1,1893 → 1,4074; razón de sumas 1,0126 → 1,2919.

**Probabilidad de venta (abril):** umbral 0,63 → 0,64; exactitud de la tarjeta 79,3 % (elegida y medida en la misma prueba interna) → 79,2 % (umbral de la validación interna, medida en la prueba interna, n = 11.703); exactitud en la validación interna 75,7 % (selección, no es resultado).

**Comparación de modelos:** solo cambia la columna Random Forest de clasificación (la de arriba). Regresión A y B sin cambio (B: RF MAE 123.997, MASE 0,5540; XGBoost 129.795; ARIMA 175.739; Prophet 197.363; media_3 246.359; ingenuo 285.081).

**Estudio de caso agrupado (4.560 decisiones, 4.155 montos):** F1 macro 0,4243 → 0,4273 (mayoritaria 0,2624 sin cambio); exactitud balanceada 0,4575 → 0,4569; exactitud 0,5682 → 0,5798; precisión Reducir 82,0 % → 80,7 %, Reforzar 41,0 % → 41,4 %, Mantener 12,0 % → 14,5 %. Mezcla: MAE 327.702 → 339.456, RMSE 936.047 → 943.216, razón 0,9071 → 0,9988. RF, ingenuo y media_3 sin cambio. F1 macro por mes 0,3599 / 0,4487 / 0,3652 / 0,4769 / 0,4329 / 0,4433 → 0,3636 / 0,4517 / 0,3573 / 0,4794 / 0,4329 / 0,4611.

**`prediccion_vs_real.xlsx` (Tabla 11, millones de COP, predicción total / real):** abril 307,6 → 363,4 / 363,2; mayo 324,6 → 358,2 / 418,7; junio 381,6 → 399,5 / 263,0; julio 333,0 → 351,7 / 348,6; agosto 325,8 → 344,8 / 372,1; septiembre 296,4 → 315,8 / 354,4; total 1.969,1 → 2.133,5 / 2.119,9 (93 % → 100,6 %). La parte por promedio (60,6 / 55,5 / 54,0 / 73,6 / 59,8 / 40,1 M) no cambia.

**¿Se venderá? (estudio de caso, 6.145 producto-mes):** 70,8 % → 70,1 % (4.348 → 4.305 aciertos); por mes 69,6 / 70,6 / 69,5 / 72,0 / 71,7 / 71,1 → 69,6 / 69,7 / 69,9 / 70,6 / 71,6 / 69,0 %. El experimento (RF balanceado) reproduce las mismas cifras.

**Grupos:** acierto ponderado agrupado 67,6 % → 68,9 % (por mes 72,1 / 64,5 / 46,6 / 74,0 / 70,6 / 72,8 → 76,4 / 68,5 / 40,3 / 74,0 / 73,1 / 73,7 %); 20 grupos con más ventas del mes anterior (criterio conocido antes del mes evaluado; antes se elegían por las ventas reales del mes evaluado: 73,4 % → 75,6 %) 74,1 % (por mes 81,0 / 71,3 / 44,2 / 81,1 / 80,2 / 79,4 %).

**Stock sugerido:** cobertura agrupada 68,4 % → 71,8 % (por mes 63,1 / 65,4 / 74,9 / 69,1 / 64,3 / 73,1 → 74,6 / 71,1 / 76,7 / 70,3 / 65,4 / 72,8 %; 5 de 6 meses en 65-75 %); unidades 71,2 % → 74,4 %; entre los que vendieron 47,3 % → 53,0 %; stock total / real 1,28 → 1,45 (más sobrestock).

**Acierto del monto por producto (producción):** 36,0 % → 36,2 % (1−WAPE 15,2 % → 12,9 %; razón 0,93 → 1,01). Las variantes del experimento no cambian.

**Dashboard (abril, corrida por defecto):** 697 productos con decisión (sin cambio); Reforzar 465 → 445, Mantener 10 → 9, Reducir 222 → 243; 348 por promedio (sin cambio); se venderán 625 → 605 de 1.045; ventas predichas 307.634.392 = 247.013.009 + 60.621.383 → 363.395.700 = 302.774.317 + 60.621.383; tarjeta F1 macro 0,45 (sin cambio), exactitud balanceada 50,4 % → 49,7 %, exactitud 66,8 % → 68,3 %, acierto ¿se venderá? 79,3 % → 79,2 %.

Se regeneraron `tuning_rf_*`, `evaluacion_modelo.txt`, `matriz_confusion.png`, `real_vs_predicho.png`, `comparacion_modelos.*`, `estudio_caso.*`, `estudio_caso_totales.png`, `prediccion_vs_real.xlsx` (copia en `../resultados/`), los experimentos de probabilidad, stock, monto, grupos, mejoras (sin cambios), sensibilidad y elegibilidad (con sus cachés borradas), `entrenar.py` y las cinco capturas `panel_*.png`.

### Limitaciones que este cambio no corrige

- La mezcla, el criterio de sesgo, el clasificador de venta (RF balanceado frente a otros), el nivel de stock y la vista por grupos se adoptaron después de mirar los seis meses del holdout en los experimentos. Los parámetros ya no se eligen con datos de reporte, pero esas decisiones de diseño sí miraron el holdout, así que el estudio de caso de esos componentes no es una prueba ciega. Una evaluación ciega requiere meses posteriores a septiembre de 2026. En la prueba interna (sin holdout) los cuatro clasificadores de venta quedan en 78,3 % a 79,8 %, sin un ganador claro.
- El clasificador binario y el regresor usan la misma regla (último 20 % por producto) sobre conjuntos de filas distintos, así que el corte puede caer en meses distintos para un mismo producto. Cada selección usa solo datos de entrenamiento.
- En `tune_classifier` la variable `cluster` sale del KMeans ajustado con toda la partición de entrenamiento y no se reajusta por pliegue (no toca la prueba interna).
- Septiembre del holdout llega al 22 (incompleto).

---

## [Figuras] Etiquetas de meses y de modelos alineadas con la tesis - 2026-10-14

- `estudio_caso_totales.png`: el eje X dice "Mes 1" ... "Mes 6*" en lugar de Abr ... Sep* (el texto y la Tabla 11 numeran los meses sin fechas). Regenerada desde `prediccion_vs_real.xlsx`.
- `comparacion_modelos.png`: la barra "Media de 3 meses" pasa a "Promedio de 3 meses". Regenerada desde `comparacion_modelos.csv`, sin reentrenar.

---

## [Sistema] Pulido mínimo del dashboard para las figuras de la tesis - 2026-10-14

Sin cambios de diseño. Las capturas `panel_*.png` se retomaron con una corrida por defecto.

- Tarjeta "Métricas de validación del modelo": F1 macro (0,45), Exactitud balanceada (50,4 %), Exactitud (66,8 %) y Acierto "¿se venderá?" (79,3 %, validación interna del modelo de probabilidad de venta). Coinciden con `evaluacion_modelo.txt`; antes mostraba las métricas ponderadas (exactitud 67 %, precisión 79 %, sensibilidad 67 %, F1 71 %) en inglés.
- Formato es-CO en todo lo visible: coma decimal, punto de miles también con 4 cifras (1.045) y "%" separado (437,9 %). Registro de ejecución con `fmt_es`/`pct_es` (`figuras.py`); el HTML usa `nEs`/`pc` con `useGrouping: 'always'`.
- Notas de Productos y Grupos: "en los seis meses de prueba" en lugar de las fechas explícitas.
- Diagnóstico del stock sugerido de 987 ($59 · 1 uds): no es un error de precio ni de unidades. Su `prob_venta` es 0,28 (cubeta [0-0,4), q70 = -8,79): en esa cubeta más del 70 % de las filas de validación no vendió, así que el cuantil 70 % de los residuos log lleva la predicción de $396.006 a $59 (cobertura esperada 72 % = no venderá). El "1 uds" es el techo de $59 / $198.000. Sin cambios de código; la cobertura de 68,4 % no cambia.
- Pruebas: `tests/test_dashboard_formato_es.py` (8); suite de 394 pruebas.

---

## [Sistema] Stock sugerido por producto - 2026-10-14

Resultado del experimento "[Experimento] Stock sugerido con nivel de servicio 70 %" (cobertura agrupada 68,4 %, 4 de 6 meses dentro de 65-75 %).

**Regla.** Paso nuevo `SalesForecastModel.agregar_stock_sugerido()`, posterior a `agregar_probabilidad_venta()` y activo por defecto (`stock_sugerido=True`, `nivel_servicio=0.70`; `run()`, `scripts/entrenar.py` y el dashboard lo ejecutan, y el formulario acepta `stock_sugerido` y `nivel_servicio`). Escribe `stock_sugerido` (COP) y `stock_sugerido_unidades` (techo del stock entre el último precio promedio conocido) para los productos con monto predicho; cota = max(expm1(log1p(pred) + q70), 0), con q70 el cuantil 70 % de log1p(real) - log1p(pred) en la validación interna del entrenamiento, por cubeta de `prob_venta` (cubetas con menos de 50 filas se fusionan; sin probabilidad, cuantil global). No cambia decisiones, montos, totales, `prob_venta` ni grupos; con `stock_sugerido=False` la salida es la de antes (probado).

- Dashboard: columna "Stock sugerido (70 %)" (COP y unidades) en Productos y en las tablas del resumen, bloque en el detalle del producto, suma por grupo en la vista Grupos, orden `sort=stock`, nota explicativa. `/api/products`, `/api/product`, `/api/summary` (tops), `/api/grupos` y el CSV traen los campos (null si el producto no tiene monto).
- Excel `prediccion_vs_real.xlsx`: columnas "Stock sugerido (70 %)", "Stock sugerido (unidades)", "Unidades reales" y "¿Cubrió la demanda?" por producto; el Resumen trae cobertura mensual (todos, los que vendieron y unidades) y la agrupada; "Cómo leer" lo explica. Coincide con el experimento: 63,1 / 65,4 / 74,9 / 69,1 / 64,3 / 73,1 %, agrupado 68,4 %.
- Pruebas: `tests/test_stock_sugerido.py` (29), `tests/test_stock_sugerido_modelo.py` (11), `tests/test_dashboard_stock.py` (12), `tests/test_prediccion_vs_real_stock.py` (10); suite de 382 pruebas.

**Corrida real (V6, mes 2026-04).** 993 productos con stock sugerido; cuantiles q70 por cubeta de `prob_venta`: [0-0,4) -8,79 (n=389), [0,4-0,6) -0,92 (n=1.441), [0,6-0,8) +0,39 (n=1.694), [0,8-1] +0,39 (n=1.620). `/api/summary` sin cambios (697, 465 / 10 / 222, 307.634.392 COP).

**Limitación.** Es un nivel de abastecimiento, no un acierto del monto: entre los productos que sí vendieron la cobertura es 47,3 %, y el stock total es 1,28 veces lo vendido.

---

## [Experimento] Stock sugerido con nivel de servicio 70 % - 2026-10-14

Objetivo del usuario: predicciones mensuales por producto confiables "alrededor del 70 %" para decidir compras. El monto puntual por producto no llega (mejor acierto medio 38 %, ver "[Experimento] Acierto del monto por producto"), así que se mide otra cosa, la que usa la teoría de inventarios: un nivel "order-up-to" X al nivel de servicio 70 % ("si abastece hasta X, la demanda del mes queda cubierta en 7 de cada 10 casos").

**Método (sin usar holdout).** cota = max(expm1(log1p(pred) + q70(r)), 0), con r = log1p(real) - log1p(pred) medido en la validación interna del entrenamiento de cada origen (partición temporal de prueba del regresor para los productos del modelo y la del clasificador de venta para los de estimación por promedio, con su media de 3 meses como predicción). Los residuos se segmentan por cubetas de `prob_venta` ([0-0,4), [0,4-0,6), [0,6-0,8), [0,8-1]; las contiguas con menos de 50 filas se fusionan; filas sin probabilidad solo cuentan en el cuantil global). También se guardan los cuantiles de 0,5 y 0,9. Unidades = techo(stock / último precio promedio conocido).

**Evaluación.** Origen rodante 2026-04..2026-09 (entrenamiento estrictamente anterior a cada mes, calibración solo con la validación interna de ese entrenamiento), universo del dashboard (6.145 producto-mes), `scripts/experimento_stock_sugerido.py` (232 s; cache `odd/stock_cache`), informe `outputs/reports/experimento_stock_sugerido.txt/.csv` y figura `experimento_stock_sugerido.png`.

- Cobertura (real <= stock sugerido, todos los productos): 63,1 / 65,4 / 74,9 / 69,1 / 64,3 / 73,1 % por mes; agrupado 68,4 % (desviación -1,6 pp respecto de 70 %); 4 de 6 meses dentro de 65-75 %. En unidades: 67,3 / 68,0 / 77,8 / 71,5 / 67,0 / 75,2 %, agrupado 71,2 %.
- Entre los productos que sí vendieron la cobertura es mucho menor: 47,3 % agrupado (42,1 a 56,7 % por mes). El 70 % incluye a los que no vendieron, para los que el sistema sugiere stock cercano a cero (cubeta de baja probabilidad de venta).
- Sobrestock: stock total / ventas reales totales 1,28 veces agrupado (1,03 a 1,99; junio, con ventas reales bajas, 1,99); mediana por producto entre los que vendieron 0,91 veces, media 2,15. Agotados (stock 0 y sí vendió): 0,2 %.
- Segmentos (agrupado): modelo 68,3 %, promedio 68,5 % (entre los que vendieron 52,5 % y 31,8 %); top 100 por predicción 86,0 %; resto 68,1 %; por probabilidad de venta [0-0,4) 80,5 %, [0,4-0,6) 67,7 %, [0,6-0,8) 68,4 %, [0,8-1] 67,0 %. Otros niveles de servicio (agrupado): 50 % -> 53,6 %, 90 % -> 88,4 %.
- Decisión: cobertura agrupada dentro de la franja 65-75 % y mayoría de meses dentro -> se integra (ver "[Sistema] Stock sugerido por producto"). El universo del dashboard ya es el top 1.900 por construcción, por eso no se reporta "top 1.900 vs resto".
- Limitación: es una cobertura de "al menos lo vendido" (puede haber sobrestock), no un acierto del monto; septiembre está incompleto (real hasta el 22).

---

## [Sistema] Vista por grupo de producto - 2026-10-14

Decisión del orquestador, dentro del objetivo del usuario de mejorar las decisiones de compra: el taller compra por línea o proveedor y el experimento "[Experimento] Predicción mensual por grupo" mostró que simplemente sumar las predicciones por producto que ya da el sistema (variante G0) acierta el 67,6 % del valor vendido por grupo (acierto ponderado agrupado, abril a septiembre de 2026), 4 de 6 meses con 70 % o más (72,1 / 64,5 / 46,6 / 74,0 / 70,6 / 72,8 %) y 73,4 % en los 20 grupos con más ventas. Ninguna variante resultó demostrablemente mejor, por eso el modelo no cambia y solo se muestra esta vista. No es una promesa de 70 %: el agrupado queda en 67,6 % y junio (las ventas reales cayeron de 492 M a 263 M) lo hunde.

**Regla.** Paso nuevo `SalesForecastModel.agregar_resumen_grupo()`, posterior a `agregar_probabilidad_venta()` y activo por defecto (`resumen_por_grupo=True`; `run()`, `scripts/entrenar.py` y el dashboard lo ejecutan). Usa el catálogo `data/raw/grupos_productos.csv` (`grupos_path`) y `ventas_forecast.grupos.resumen_grupos()`: por grupo, productos con monto predicho (modelo y promedio), cuántos "se venderán", suma de `ventas_predichas`, ventas del mes anterior de esos mismos productos, variación y porcentaje del total; los productos sin grupo van en "Sin grupo". Solo lee los resultados: no cambia ninguna predicción, decisión ni total (probado: con la opción apagada `results` y el CSV son idénticos y no hay resumen).

- Dashboard: vista "Grupos" en la barra lateral (tabla ordenada por predicción, buscador y nota con el acierto medido). Un clic en un grupo abre "Productos" filtrado por ese grupo (chip "Grupo: ..." para quitarlo; enlace directo `/#grupos`). `/api/grupos` devuelve la tabla y `pred_total`; `/api/products` acepta `grupo=<nombre>` y devuelve la columna `grupo`.
- Excel `prediccion_vs_real.xlsx`: hoja "Grupos" con un resumen por mes (predicción, real, acierto ponderado y número de grupos, más la fila "Agrupado") y el detalle por mes y grupo (predicción, real, diferencia, acierto %); "Cómo leer" explica la hoja. El acierto ponderado reproduce al experimento: 72,1 / 64,5 / 46,6 / 74,0 / 70,6 / 72,8 % por mes y 67,6 % agrupado (482 grupo-mes).
- Pruebas: `tests/test_resumen_grupo.py` (13), `tests/test_dashboard_grupos.py` (9) y 3 nuevas en `tests/test_prediccion_vs_real.py`; suite de 320 pruebas.

**Corrida real (V6, mes 2026-04).** 83 grupos y 1.045 productos; la suma de los grupos es 307.634.392 COP, igual a `pred_total` de `/api/summary` (que no cambió: 697 productos con decisión, 465 / 10 / 222). Mayores grupos por predicción: BANDAS INCOLBEST 78,6 M (25,6 %), RETEN VICTOR SNA 32,7 M, GRASA 31,1 M, BANDAS MACKBLOCK 20,0 M, RODILLOS 18,5 M.

---

## [Experimento] Predicción mensual por grupo - 2026-10-14

Experimento sin cambios en producción (modelo, dashboard y `entrenar.py` intactos). Meta del usuario: un pronóstico mensual de ~70 % que sirva para decidir compras; por producto no se alcanza (mejor acierto medio 38 %, ver "[Experimento] Acierto del monto por producto"), así que se probó el nivel de **grupo de producto** (`PRODUCTO.IDGRUPO`, el nivel en que el taller compra por línea o proveedor). Script `scripts/experimento_grupos.py`, ayudantes puros `src/ventas_forecast/grupos.py` (22 pruebas), catálogo `data/raw/grupos_productos.csv` (6.395 productos, 144 nombres de grupo; se regenera con `sql/extraer_grupos.py`; sin datos de clientes), informe `outputs/reports/experimento_grupos.txt/.csv` y figura `outputs/figures/experimento_grupos.png`.

**Diseño.** Mismo origen rodante del estudio de caso (meses 2026-04 a 2026-09; septiembre llega al 22) y entrenamiento solo con datos anteriores a cada mes. Universo = productos que muestra el dashboard en cada origen (modelo con monto + estimación por promedio), sumados por grupo (77 a 83 grupos por mes; ningún producto sin grupo). También se mide contra "todos los productos del grupo" (incluye los que el sistema no predice). Métrica principal: acierto ponderado por valor 1−WAPE entre grupos (Σ|R−P| / ΣR), agrupado y por mes; además acierto medio por grupo (1 − |R−P|/R recortado, R > 0), porcentaje de grupos con acierto ≥ 70 %, los 20 grupos con más ventas y razón de sumas. Variantes: **G0** suma de las predicciones por producto de producción; **G1** media de 3 meses del grupo; **G2** modelo directo (RF; G2x con XGBoost) sobre la serie mensual del grupo con rezagos 1/2/3/6/12, medias 3/6, mes seno/coseno, productos activos y facturas, objetivo log1p y mezcla con G1 cuyo peso sale de una validación interna (últimos 3 meses conocidos, modelo que no los ve); **G3** mezcla de G0 y G2 (G3b: con G1) con el peso que minimiza el error en los meses ya cerrados (0,5 sin historia; G3f: 50/50 fijo); **G4** G0 reconciliada al total del grupo de G2 (G4b: de G1). Nada se ajusta con el holdout. La elección de "la mejor variante" al leer la tabla sí mira el holdout y es optimista.

**Verificación.** G0 reproduce 67,6 % agrupado (acierto medio por grupo 49,1 %; el cálculo previo del orquestador daba 49,2 %; la diferencia de una décima no se investigó). Las ventas del último mes de las series por grupo coinciden con las del modelo de cada origen.

**Acierto ponderado por valor (universo), por mes y agrupado.**

| Variante | Abr | May | Jun | Jul | Ago | Sep* | Agrupado | Meses ≥ 70 % |
|---|---|---|---|---|---|---|---|---|
| G0: suma de productos (producción) | 72,1 % | 64,5 % | 46,6 % | 74,0 % | 70,6 % | 72,8 % | 67,6 % | 4 de 6 |
| G1: media de 3 meses del grupo | 76,4 % | 70,3 % | 34,0 % | 72,0 % | 70,9 % | 72,4 % | 67,6 % | 5 de 6 |
| G2: modelo directo (RF) | 76,4 % | 70,3 % | 32,5 % | 76,0 % | 73,2 % | 72,3 % | 68,4 % | 5 de 6 |
| G2x: modelo directo (XGBoost) | 76,4 % | 70,3 % | 34,0 % | 71,6 % | 73,9 % | 71,6 % | 67,9 % | 5 de 6 |
| G3: mezcla G0 y G2 | 76,4 % | 69,3 % | 32,5 % | 75,1 % | 73,4 % | 73,2 % | 68,3 % | 4 de 6 |
| G3b: mezcla G0 y G1 | 76,4 % | 69,3 % | 34,0 % | 73,9 % | 73,2 % | 73,4 % | 68,2 % | 4 de 6 |
| G3f: mezcla 50/50 G0 y G2 | 76,4 % | 67,6 % | 39,5 % | 75,1 % | 73,4 % | 73,4 % | 68,8 % | 4 de 6 |
| G4: G0 reconciliada a G2 | 76,4 % | 70,3 % | 32,5 % | 76,0 % | 73,2 % | 72,3 % | 68,4 % | 5 de 6 |
| G4b: G0 reconciliada a G1 | 76,4 % | 70,3 % | 34,0 % | 72,0 % | 70,9 % | 72,4 % | 67,6 % | 5 de 6 |

(*) Septiembre incompleto. Otras métricas agrupadas (G0 / G1 / G2 / G3f): acierto medio por grupo 49,1 / 47,3 / 47,3 / 48,3 %; grupos con acierto ≥ 70 % 33,8 / 32,4 / 34,3 / 33,1 %; 20 grupos con más ventas 73,4 / 74,5 / 74,7 / 74,9 %; razón de sumas 0,93 / 1,08 / 1,05 / 0,99. Intervalos de 95 % (remuestreo de grupo-meses) del acierto agrupado: 60 a 74 % para todas las variantes, es decir, ninguna se distingue de G0 ni del 70 %.

**Todos los productos del grupo.** G0 baja a 60,1 % (el universo cubre 71 a 80 % del real del grupo) y la razón de sumas es 0,71; G1 65,4 %, G2 66,0 % y G2x 65,2 % (3, 4 y 2 meses de 6 sobre 70 %).

**Productos (G4).** Reconciliar a G2 o G1 no ayuda por producto: 1−WAPE por producto 15,2 % (G0) frente a 11,4 % (G4) y 10,3 % (G4b); acierto medio 36,0 / 35,7 / 35,6 %.

**Resultado.** Ninguna variante llega a 70 % agrupado en el universo; la mejor (G3f, 68,8 %; G2 68,4 % con el peso elegido por validación) queda a 1 o 2 puntos. Por mes se supera 70 % en 4 a 5 de los 6 meses, pero junio rompe el agrupado: el real cayó a 263 M frente a 492 M de mayo (un salto que ningún método anticipa, solo G0 se defiende mejor con 46,6 %). Descriptivo y a posteriori, sin junio el agrupado sería 70,6 % (G0), 72,3 % (G1), 73,5 % (G2) y 73,0 % (G3f); no se usó para elegir nada. Por grupo individual el acierto sigue siendo bajo (medio 47 a 49 %, solo un tercio de los grupos llega a 70 %); lo que sí alcanza ~70 % es el acierto ponderado por valor, dominado por los grupos grandes (los 20 con más ventas: 73 a 75 %).

**Recomendación.** No se integra al sistema: la meta de 70 % agrupado no se cumple y la ganancia de G2/G3 sobre la suma de productos actual (G0, 67,6 %) no es demostrable (diferencia 0,8 a 1,2 puntos, intervalos superpuestos). Si se quiere una guía de compra por línea, la suma de la predicción por producto agrupada por grupo ya ofrece ~68 % ponderado (~73 % en los 20 grupos con más ventas) sin cambiar el modelo, siempre advirtiendo que el acierto por grupo pequeño es bajo y que un mes con un salto de demanda no se anticipa.

## [Sistema] Probabilidad de venta en el dashboard - 2026-10-14

Decisión del orquestador, dentro del objetivo del usuario de mejorar las decisiones de compra: llevar a producción el clasificador binario "¿tendrá ventas el mes siguiente?" del experimento "[Experimento] Probabilidad de venta (meta 70 %)" (Random Forest balanceado: exactitud agrupada 70,8 %, balanceada 69,8 %, AUC 0,773, frente a 62,5 % de la regla "vendió el mes anterior"). Aporta una probabilidad y un orden de los productos (AUC 0,77) que la regla simple no da; frente a la regla "fracción histórica > 0,5" (70,3 %) la diferencia no es demostrable, como dice el experimento. Los montos no cambian: el experimento "Acierto del monto por producto" no encontró una variante que valga la pena desplegar.

**Regla.** Paso nuevo `SalesForecastModel.agregar_probabilidad_venta()`, posterior a `estimar_por_promedio()` y activo por defecto (`probabilidad_venta=True`; `run()`, `scripts/entrenar.py` y el dashboard lo ejecutan). Usa `ventas_forecast.sale_probability` (variables, marco de entrenamiento y selector de umbral) con la misma configuración del experimento: RandomForest de 300 árboles, `max_depth=12`, `min_samples_leaf=5`, `class_weight="balanced"`. Se entrena solo con la partición temporal de entrenamiento (80 % por producto) y el umbral se elige con la validación interna (nunca con el mes a predecir). Escribe en los resultados `prob_venta` (0 a 1) y `se_vendera` ("Sí" / "No") para todos los productos, los del modelo y los de estimación por promedio, y guarda `sale_threshold` y `sale_val_accuracy`.

- El clasificador de 3 clases, el regresor, la mezcla y la estimación por promedio no cambian (probado: con la opción encendida las demás columnas, `blend_weight` y las predicciones del clasificador y del regresor son iguales). Con `probabilidad_venta=False` la salida y el CSV son los de antes (probado).
- Dashboard: `/api/summary` devuelve `n_se_vendera`, `n_con_prob_venta`, `sale_threshold` y `sale_val_accuracy`; KPI "Probablemente se venderán: N de M productos". `/api/products` acepta `venta=Sí|No` y `sort=venta_prob` y devuelve `prob_venta` y `se_vendera`; el catálogo y las tablas del resumen tienen las columnas "Prob. de venta" y "Se venderá", el catálogo el filtro "Se venderá / No se venderá", y el detalle muestra la probabilidad y "Se venderá: Sí/No" también para los productos por promedio. El CSV descargable trae ambas columnas. Con resultados antiguos (sin las columnas) todo se comporta como antes.
- Reporte `prediccion_vs_real.xlsx`: columnas "Probabilidad de venta", "¿Se venderá? (predicción)", "¿Se vendió?" (real mayor a 0) y "¿Acertó venta?" para todos los productos; la hoja "Resumen" agrega "Productos evaluados (venta)", "Ventas acertadas", "% acierto de venta" (productos del modelo con monto más los de promedio, el universo del experimento) y "% acierto de venta (todos)".
- Pruebas: `tests/test_probabilidad_venta.py` (9), `tests/test_dashboard_prob_venta.py` (10) y 5 nuevas en `tests/test_prediccion_vs_real.py`; suite de 269 pruebas.

**Corrida real (V6, mes 2026-04).** Umbral 0,63 y exactitud de validación interna 79,3 %; 625 de 1.045 productos "se venderán". Los totales no cambian: `/api/summary` da 307.634.392 COP (modelo 247.013.009 + promedio 60.621.383) y las decisiones son las de siempre (465 Reforzar, 10 Mantener, 222 Reducir).

**Acierto de venta contra lo real** (hoja "Resumen", abril a septiembre de 2026, origen rodante): 69,6 %, 70,6 %, 69,5 %, 72,0 %, 71,7 % y 71,1 % por mes; agrupado 4.348 de 6.145 producto-mes (70,8 %). Coincide con el RF balanceado del experimento (universo del dashboard) mes a mes y agrupado. Incluyendo los productos del modelo sin monto ("todos"): 69,9 %, 70,0 %, 69,3 %, 72,0 %, 71,4 % y 71,2 %. La exactitud de validación interna (79,3 % en abril) se mide sobre otras filas (la validación temporal del entrenamiento) y no es comparable con el acierto de los meses evaluados; el umbral se elige con ella y nunca con el mes predicho. Septiembre está incompleto.

---

## [Experimento] Acierto del monto por producto - 2026-10-14

Experimento sin cambios en producción (modelo, dashboard y `entrenar.py` intactos). Meta del usuario: acercar a ~70 % el acierto del monto predicho frente al real por producto. Script `scripts/experimento_monto_producto.py`, ayudantes puros `src/ventas_forecast/monto_producto.py` (23 pruebas), informe `outputs/reports/experimento_monto_producto.txt/.csv` y figura `outputs/figures/experimento_monto_producto.png`.

**Diseño.** Mismo origen rodante del estudio de caso y mismo universo del dashboard (productos del modelo con monto + productos por promedio). Para cada horizonte (1, 2 y 3 meses) se entrena con el objetivo a ese horizonte (suma de los h meses siguientes) usando solo datos anteriores al primer mes del bloque. Bloques sin traslape: mensual abril a septiembre; bimestral abr-may, jun-jul, ago-sep; trimestral abr-jun, jul-sep (septiembre incompleto, llega al 22). Potencia Tweedie {1,2; 1,5; 1,8}, modelo condicional (RF o XGB), umbral de P(venta), pesos de mezcla y factores de corrección se eligen solo con la validación interna (último 20 % de cada producto). Las variantes señaladas como "mejores" se escogen mirando el holdout, por lo que son optimistas.

Variantes: **Base** (producción x h); **Control** (pipeline de producción reentrenado a h meses; reproduce la Base mensual: 35,9 % / 18,4 % / 15,1 % frente a 36,0 % / 18,4 % / 15,2 %); **C** (variables de probabilidad de venta en el RF de producción); **B** (XGBoost `reg:tweedie` en pesos); **A0** (monto condicional), **A1** (P(venta) x monto condicional), **A2** (monto condicional si P >= umbral, si no 0); **+M** (mezcla con media_3); **+D** (corrección por deciles del monto predicho, que maximiza el acierto o corrige la suma; A+D equivale a A+C+D porque A ya usa las variables enriquecidas). Métricas: acierto medio = media de 1 − |real − pred| / real recortado a [0, 1] sobre productos con real > 0; ±30 % y ±50 % sobre esos productos; 1−WAPE y razón de sumas sobre todo el universo.

**Agrupado mensual.**

| Variante | Acierto medio | ±30 % | ±50 % | 1−WAPE | Pred/real | MAE (COP) | n |
|---|---|---|---|---|---|---|---|
| Base (producción) | 36,0 % | 18,4 % | 33,2 % | 15,2 % | 0,93 | 292.485 | 6.145 |
| Media de 3 meses | 36,0 % | 19,6 % | 34,2 % | 10,1 % | 1,08 | 310.138 | 6.145 |
| Control: RF de producción | 35,9 % | 18,4 % | 33,3 % | 15,1 % | 0,94 | 292.775 | 6.145 |
| C: RF enriquecido | 36,1 % | 18,4 % | 33,4 % | 15,4 % | 0,93 | 291.961 | 6.145 |
| B: XGBoost Tweedie | 34,4 % | 18,5 % | 32,3 % | 10,2 % | 0,74 | 309.882 | 6.145 |
| A0: monto condicional | 36,9 % | 21,7 % | 37,1 % | 4,4 % | 1,17 | 329.841 | 6.145 |
| A1: P(venta) × monto | 38,0 % | 20,8 % | 35,9 % | 17,8 % | 0,93 | 283.720 | 6.145 |
| A2: monto si P ≥ umbral | 28,0 % | 16,5 % | 28,4 % | 21,4 % | 0,87 | 271.040 | 6.145 |
| B+M: Tweedie y media 3 | 36,5 % | 19,8 % | 35,0 % | 11,3 % | 1,05 | 306.031 | 6.145 |
| A1+M: A1 y media 3 | 38,3 % | 21,0 % | 36,4 % | 14,8 % | 0,99 | 293.876 | 6.145 |
| C+D (acierto) | 37,1 % | 18,6 % | 33,4 % | 16,7 % | 0,75 | 287.408 | 6.145 |
| C+D (suma) | 37,0 % | 20,7 % | 36,1 % | 14,1 % | 0,96 | 296.461 | 6.145 |
| A0+D (acierto) | 37,8 % | 19,6 % | 35,6 % | 12,0 % | 0,89 | 303.677 | 6.145 |
| A0+D (suma) | 38,2 % | 21,1 % | 36,2 % | 11,7 % | 0,94 | 304.728 | 6.145 |
| A1+D (acierto) | 38,2 % | 20,2 % | 36,0 % | 18,6 % | 0,80 | 280.968 | 6.145 |
| A1+D (suma) | 36,7 % | 20,1 % | 35,6 % | 11,9 % | 1,06 | 303.913 | 6.145 |

**Agrupado bimestral** ("Suma mensual rodante" usa orígenes posteriores al del bloque; es solo referencia).

| Variante | Acierto medio | ±30 % | ±50 % | 1−WAPE | Pred/real | MAE (COP) | n |
|---|---|---|---|---|---|---|---|
| Base (producción) | 37,8 % | 20,5 % | 36,6 % | 27,4 % | 0,96 | 506.327 | 3.038 |
| Suma mensual rodante (ref.) | 42,4 % | 23,8 % | 42,3 % | 34,9 % | 0,94 | 454.243 | 3.038 |
| Media de 3 meses | 37,6 % | 21,3 % | 37,4 % | 23,1 % | 1,10 | 536.201 | 3.038 |
| Control: RF de producción | 37,2 % | 20,4 % | 36,1 % | 26,7 % | 1,03 | 511.421 | 3.038 |
| C: RF enriquecido | 37,7 % | 21,2 % | 37,3 % | 29,4 % | 1,03 | 492.546 | 3.038 |
| B: XGBoost Tweedie | 30,5 % | 18,2 % | 31,2 % | -11,5 % | 1,36 | 777.764 | 3.038 |
| A0: monto condicional | 37,9 % | 24,1 % | 39,5 % | 22,9 % | 1,13 | 537.538 | 3.038 |
| A1: P(venta) × monto | 39,2 % | 23,6 % | 38,9 % | 29,0 % | 0,99 | 495.491 | 3.038 |
| A2: monto si P ≥ umbral | 33,2 % | 21,4 % | 34,8 % | 27,9 % | 1,01 | 502.894 | 3.038 |
| B+M: Tweedie y media 3 | 38,3 % | 21,9 % | 38,4 % | 25,7 % | 1,08 | 518.031 | 3.038 |
| A1+M: A1 y media 3 | 38,9 % | 22,3 % | 39,0 % | 25,9 % | 1,06 | 516.504 | 3.038 |
| C+D (acierto) | 38,6 % | 21,0 % | 36,4 % | 32,2 % | 0,80 | 472.965 | 3.038 |
| C+D (suma) | 38,0 % | 23,5 % | 38,8 % | 29,6 % | 1,02 | 491.288 | 3.038 |
| A0+D (acierto) | 39,3 % | 23,0 % | 38,9 % | 25,4 % | 0,92 | 520.043 | 3.038 |
| A0+D (suma) | 37,5 % | 23,3 % | 38,1 % | 20,3 % | 1,18 | 555.924 | 3.038 |
| A1+D (acierto) | 39,2 % | 22,6 % | 38,4 % | 28,9 % | 0,85 | 495.928 | 3.038 |
| A1+D (suma) | 35,1 % | 21,6 % | 37,3 % | 19,3 % | 1,25 | 563.066 | 3.038 |

**Agrupado trimestral.**

| Variante | Acierto medio | ±30 % | ±50 % | 1−WAPE | Pred/real | MAE (COP) | n |
|---|---|---|---|---|---|---|---|
| Base (producción) | 38,0 % | 20,0 % | 36,5 % | 34,3 % | 0,91 | 684.140 | 2.035 |
| Suma mensual rodante (ref.) | 48,4 % | 28,7 % | 51,7 % | 48,3 % | 0,93 | 537.848 | 2.035 |
| Media de 3 meses | 37,1 % | 20,0 % | 36,4 % | 29,5 % | 1,07 | 733.416 | 2.035 |
| Control: RF de producción | 35,4 % | 20,2 % | 35,5 % | 26,1 % | 1,17 | 769.386 | 2.035 |
| C: RF enriquecido | 34,6 % | 20,2 % | 33,8 % | 28,2 % | 1,15 | 747.770 | 2.035 |
| B: XGBoost Tweedie | 33,7 % | 22,1 % | 36,4 % | 27,8 % | 1,16 | 750.997 | 2.035 |
| A0: monto condicional | 35,9 % | 23,6 % | 36,6 % | 24,2 % | 1,28 | 788.488 | 2.035 |
| A1: P(venta) × monto | 38,1 % | 23,9 % | 38,8 % | 29,0 % | 1,17 | 738.808 | 2.035 |
| A2: monto si P ≥ umbral | 33,7 % | 22,4 % | 34,5 % | 25,9 % | 1,22 | 771.093 | 2.035 |
| B+M: Tweedie y media 3 | 39,0 % | 22,0 % | 39,9 % | 33,3 % | 1,07 | 693.933 | 2.035 |
| A1+M: A1 y media 3 | 38,3 % | 20,6 % | 38,3 % | 33,2 % | 1,07 | 695.152 | 2.035 |
| C+D (acierto) | 38,1 % | 20,2 % | 36,6 % | 36,7 % | 0,80 | 658.385 | 2.035 |
| C+D (suma) | 34,3 % | 20,4 % | 34,3 % | 30,1 % | 1,11 | 727.978 | 2.035 |
| A0+D (acierto) | 38,9 % | 22,9 % | 39,6 % | 30,7 % | 1,00 | 721.215 | 2.035 |
| A0+D (suma) | 35,6 % | 23,6 % | 37,4 % | 20,4 % | 1,34 | 828.114 | 2.035 |
| A1+D (acierto) | 39,4 % | 23,6 % | 40,9 % | 33,2 % | 0,97 | 695.224 | 2.035 |
| A1+D (suma) | 32,4 % | 22,7 % | 34,9 % | 18,7 % | 1,37 | 846.006 | 2.035 |

**Exactitud de "vende o no"** (P >= umbral de cada bloque): mensual 70,6 % (regla "vendió el mes anterior" 62,5 %; 60,0 % de los productos vende), bimestral 77,6 % (regla 68,3 %; 77,7 % vende), trimestral 83,8 % (regla 67,0 %; 84,8 % vende). En bimestral y trimestral el clasificador no supera a "siempre vende".

**Por bloque y segmentos.** Las tablas por bloque y por segmento (top 100 del real, venden todos los meses, resto) están en el informe. Ejemplo mensual, top 100 (n = 600): Base 41,1 % de acierto medio y 43,1 % de 1−WAPE, razón 0,64; A1+M 43,6 % / 44,7 %, razón 0,66. Los productos que no venden todos los meses (resto) tienen en bimestral y trimestral un 1−WAPE negativo en la Base (−121,4 % y −57,3 %) porque se sobreestiman.

**¿Se alcanza el 70 %?** No. Ninguna variante, horizonte, bloque ni segmento llega a 70 % en acierto medio, ±30 %, ±50 % ni 1−WAPE. Máximos en cualquier combinación: acierto medio 56,5 % (B, trimestral, abr-jun, top 100, n = 100), ±30 % 43,0 %, ±50 % 59,0 %, 1−WAPE 59,5 % (n = 335 y 100). El 70 % solo se logra en la pregunta "¿vende o no?" del experimento anterior, no en el monto.

**Diferencia del mejor frente a la Base** (acierto medio, pareada, IC 95 % orientativo): mensual A1+M +2,26 puntos (+1,53 a +3,00); bimestral A0+D +1,48 (−0,01 a +2,97); trimestral A1+D +1,44 (−0,44 a +3,32). Solo la mejora mensual se distingue del ruido; es pequeña (36,0 % a 38,3 %).

**Hallazgos.** (1) Enriquecer el RF de producción con las variables de probabilidad no cambia el resultado (36,1 % frente a 36,0 % mensual). (2) Tweedie en pesos no mejora y es inestable en bimestral (1−WAPE −11,5 %). (3) El mejor aporte mensual es multiplicar el monto condicional por P(venta) (A1: 38,0 %, 1−WAPE 17,8 %). (4) La corrección por tamaño (D) mejora a veces el acierto medio a costa de subpredecir (razón de sumas 0,75 a 0,97 en las variantes de acierto) y no es estable entre bloques. (5) Anular el monto cuando P < umbral (A2) empeora el acierto medio (28,0 % mensual) porque los falsos negativos quedan en 0, aunque baja el MAE. (6) Sumar más meses ayuda al 1−WAPE de la Base (15,2 % mensual, 27,4 % bimestral, 34,3 % trimestral) pero el acierto medio casi no sube (36,0 %, 37,8 %, 38,0 %); el acierto por producto está limitado por la intermitencia de las ventas.

**Conclusión y recomendación.** La meta de ~70 % por producto no es alcanzable con estos datos y variantes; el techo observado ronda 38 a 39 % de acierto medio. No se recomienda cambiar el modelo de producción por una mejora de 2 puntos. Si se quiere una mejora de bajo riesgo, A1+M (probabilidad de venta x monto condicional mezclado con media_3) en el horizonte mensual es la única que se separa del ruido; para planificar, el trimestre da el mejor 1−WAPE (Base 34,3 %; C+D 36,7 %) y conviene presentar el monto agregado (por categoría o total) en vez de por producto. Cualquier adopción debe validarse en meses nuevos, porque la elección de la "mejor" variante se hizo sobre el mismo holdout.

# 📝 CHANGELOG - Control de Versiones

## [Experimento] Probabilidad de venta (meta 70 %) - 2026-10-14

Experimento sin cambios en producción (modelo, dashboard y `entrenar.py` intactos). Pregunta: ¿el producto tendrá ventas el mes siguiente (sí/no)? Meta del usuario: 70 % de exactitud en meses reales no vistos. Código: `src/ventas_forecast/sale_probability.py` (variables, marco de entrenamiento, umbral, métricas; `tests/test_sale_probability.py`, 20 pruebas; suite de 222 OK) y `scripts/experimento_probabilidad_venta.py`. Informe: `outputs/reports/experimento_probabilidad_venta.txt`, `.csv` y `_calibracion.csv`; figura `outputs/figures/experimento_probabilidad_venta.png` (`outputs/` no se versiona).

**Diseño.** Origen rodante igual al del estudio de caso (2026-04 a 2026-09; entrenamiento con la V6 más el holdout anterior a cada mes). Objetivo `vende` = ventas del mes siguiente > 0 sobre el calendario completo; el marco incluye filas con mes actual en cero si hubo ventas en los últimos 6 meses; partición temporal 80/20 por producto y el 20 % (validación interna) fija el umbral que maximiza la exactitud. Nada se eligió con el holdout. Variables nuevas, sin mirar al futuro (probado): meses desde la última venta, meses con venta en los últimos 3/6/12, rezago 12, rezagos 1-3 de FRECUENCIA y su media, edad y fracción histórica de meses con venta. Modelos: Random Forest (sin pesos y balanceado), XGBoost y regresión logística; reglas: vendió el mes anterior, ≥ 2 de los últimos 3 meses, fracción histórica > 0,5. Universo (a): el del dashboard (modelo con monto + estimación por promedio): 6.145 producto-mes, 60,0 % vende y la regla "vendió el mes anterior" acierta 62,5 %, como en la línea base. Universo (b): productos con ≥ 1 venta en los 6 meses previos (14.440 producto-mes, 35,4 % vende).

**Exactitud agrupada, universo (a).**

| Modelo / regla | Exactitud | Ex. balanceada | AUC |
|---|---|---|---|
| Random Forest | 70,2 % | 69,4 % | 0,7724 |
| RF balanceado | 70,8 % | 69,8 % | 0,7732 |
| XGBoost | 70,3 % | 69,8 % | 0,7681 |
| Regresión logística | 70,4 % | 69,0 % | 0,7676 |
| Regla: vendió el mes anterior | 62,5 % | 59,3 % | - |
| Regla: ≥2 de los últimos 3 | 65,4 % | 62,3 % | - |
| Regla: fracción histórica > 0,5 | 70,3 % | 69,2 % | - |

**Exactitud por mes, universo (a).**

| Modelo / regla | 2026-04 | 2026-05 | 2026-06 | 2026-07 | 2026-08 | 2026-09 |
|---|---|---|---|---|---|---|
| Random Forest | 68,8 % | 70,3 % | 70,2 % | 70,8 % | 70,9 % | 70,3 % |
| RF balanceado | 69,6 % | 70,6 % | 69,5 % | 72,0 % | 71,7 % | 71,1 % |
| XGBoost | 68,4 % | 72,0 % | 69,7 % | 70,3 % | 71,3 % | 70,3 % |
| Regresión logística | 70,7 % | 68,9 % | 70,5 % | 70,6 % | 71,7 % | 70,2 % |
| Regla: vendió el mes anterior | 62,4 % | 62,7 % | 62,1 % | 62,6 % | 66,1 % | 59,1 % |
| Regla: ≥2 de los últimos 3 | 64,9 % | 65,9 % | 63,7 % | 65,2 % | 69,9 % | 62,7 % |
| Regla: fracción histórica > 0,5 | 71,3 % | 69,3 % | 70,0 % | 70,5 % | 70,6 % | 69,9 % |
| % real que vende | 63,6 % | 59,7 % | 58,0 % | 59,5 % | 65,2 % | 54,3 % |

**Universo (b), agrupado.** Random Forest 77,8 %, RF balanceado 78,0 %, XGBoost 77,2 %, regresión logística 77,9 %; reglas 66,9 % / 73,8 % / 75,0 %. Los 4 modelos superan el 70 % en los 6 meses.

**Segmentos del universo (a).** Productos del modelo (n = 4.155): 72,9 % a 73,7 % los modelos; productos por promedio (n = 1.990): 64,4 % a 64,7 %. En (b), top 1.900: 73,1 % a 73,9 %; resto: 85,1 % a 85,9 % (14,9 % vende, predice casi siempre "no").

**Conclusión (honesta).** La meta se cumple en el agrupado del universo (a) con los 4 modelos (70,2 % a 70,8 %), pero por poco: el IC 95 % del mejor (RF balanceado, 70,76 %) va de 69,62 % a 71,89 % y por mes la meta se cumple en 4 a 5 de 6 meses según el modelo (abril la incumple casi todos). Además la regla simple "fracción histórica de meses con venta > 0,5" logra 70,3 % sin entrenar: el mejor modelo la supera en 0,50 puntos (IC 95 %: -0,25 a 1,25), es decir, sin diferencia demostrable. El AUC es ~0,77 y la calibración sobreestima la probabilidad en los deciles medios. El límite está en los productos por promedio (64,6 %), cuya venta es intermitente. No hizo falta la ronda de mejora (el agrupado ya era ≥ 70 %). Recomendación: si se lleva al sistema, una probabilidad de venta basada en recencia/frecuencia es defendible, pero se debe presentar como ~70 % (no más) y comparar con la regla simple; en el universo (b) la exactitud sube a ~78 %.

---

## [Sistema] Estimación por promedio para productos sin venta el último mes - 2026-10-13

Decisión del usuario tras el experimento "Elegibilidad ampliada": el clasificador es peor que la clase mayoritaria en esos productos (exactitud 0,4747 frente a 0,6776, Reforzar predicho 66,3 % frente a 29,6 % real), así que NO reciben decisión del modelo; pero su monto sí se estima, porque el promedio de 3 meses cubre mejor el total.

**Regla.** Paso nuevo `SalesForecastModel.estimar_por_promedio()`, posterior a `predict_next_month()` y activo por defecto (`estimacion_por_promedio=True`; `run()`, `scripts/entrenar.py` y el dashboard lo ejecutan). Un producto entra si cumple a la vez: sin ventas en el último mes, `media_3_actual > 0` (media de los últimos 3 meses incluido el último mes en cero, es decir, vendió en alguno de los 2 meses anteriores), al menos `min_months` meses con ventas y pertenencia a los `top_n_products` del regresor (como el monto de los demás). Su `ventas_predichas` es `media_3`; `ventas_predichas_rf`, `prob_crecimiento` y `prob_reduccion` quedan vacías y su decisión es "Sin recomendación (estimación por promedio)".

- El entrenamiento del clasificador y del regresor no cambia: no se agregan sus filas y no se usa `elegibilidad_ampliada` (sigue en `False`). Con `estimacion_por_promedio=False` el resultado y el CSV son los de antes (probado).
- Resultados y CSV: columna nueva `origen_estimacion` ("modelo" / "promedio"); sin el paso la columna no existe y el dashboard trata todo como "modelo".
- Dashboard: `/api/summary` devuelve `pred_total` = total del modelo + total por promedio, más `pred_total_modelo`, `pred_total_promedio`, `n_promedio` y `con_pred_modelo` (se mantienen `pred_total_rf` y `blend_weight`). `total`, Reforzar/Mantener/Reducir, la dona, el promedio de probabilidad y las tablas de probabilidad cuentan solo los productos del modelo; un KPI nuevo muestra "Sin recomendación: M productos". El KPI "Ventas predichas" muestra el total y la nota "Modelo combinado: $X (N productos) · Estimación por promedio: $Y (M productos)". El historial cubre los mismos productos que el total (modelo con monto + promedio). El catálogo tiene el filtro "Sin recomendación (estimación por promedio)"; el detalle de esos productos dice "Estimación por promedio de los últimos 3 meses" y no muestra probabilidad.
- Pruebas: `tests/test_estimacion_promedio.py` (10) y `tests/test_dashboard_promedio.py` (9); suite de 193 pruebas (202 con la lista de predicción contra realidad).

**Corrida real (V6, mes 2026-04).** 697 productos con decisión (645 con monto) y 348 por promedio; modelo 247,0 M + promedio 60,6 M = 307,6 M. Las decisiones son las de siempre (465 Reforzar, 10 Mantener, 222 Reducir). Frente a las ventas reales de abril: productos del modelo con monto 302,4 M, productos por promedio 60,8 M, los 1.900 con más ventas 416,0 M; la cobertura del real de los 1.900 pasa de 0,594 (solo modelo, 247,0 / 416,0) a 0,740 (307,6 / 416,0) y la razón predicho / real de los mismos productos es 0,847 (307,6 / 363,2). El total del modelo no cambia.

### Lista de predicción contra realidad por producto

`scripts/reporte_prediccion_vs_real.py` (necesita `openpyxl`, en `requirements-comparacion.txt`) genera `outputs/reports/prediccion_vs_real.xlsx` y `prediccion_vs_real_abril.csv` (`;`, UTF-8 con BOM). Hojas: "Cómo leer", "Resumen" (totales por mes, cobertura, % de decisiones acertadas y MAE), "Abril 2026" (el mes que predice el dashboard, entrenado como `entrenar.py`) y una hoja por mes del estudio de caso (2026-04 a 2026-09, origen rodante; septiembre incompleto). Columnas: código, descripción, origen, decisión, probabilidad, ventas del mes anterior, predicción combinada, solo modelo, real, diferencia, error %, clase real y acierto. Clase real: crecimiento mayor a 15 % Reforzar, caída mayor a 10 % o cero Reducir, en otro caso Mantener (en blanco para los productos por promedio). Decisiones acertadas por mes: 2026-04 360/697 (51,6 %), 05 424/760 (55,8 %), 06 390/764 (51,0 %), 07 443/724 (61,2 %), 08 449/759 (59,2 %), 09 525/856 (61,3 %). La hoja de abril coincide con `/api/summary` de la corrida real (247.013.009 + 60.621.383 = 307.634.392 COP).

---

## [Experimento] Elegibilidad ampliada - 2026-10-13

Experimento sin cambios en el comportamiento de producción: `SalesForecastModel` gana el parámetro opcional `elegibilidad_ampliada` (por defecto `False`; con `False` el conjunto elegible, las etiquetas, la partición y las predicciones son los de siempre, verificado con pruebas) y los ayudantes puros viven en `src/ventas_forecast/eligibility.py`. El modelo, el dashboard y `entrenar.py` no cambian. Informe: `outputs/reports/experimento_elegibilidad.txt/.csv`; figura `outputs/figures/experimento_elegibilidad.png` (`outputs/` no se versiona; se regenera con `scripts/experimento_elegibilidad.py`).

Problema: hoy solo se predice un producto si vendió en el último mes. En abril de 2026 los productos del top 1900 sin ventas en marzo quedaban fuera, aunque muchos vendieron en abril.

**Regla de la variante.** Además de los productos con ventas en el último mes, se entrenan y predicen los que NO vendieron el último mes pero tienen `media_3 > 0` (media de los últimos 3 meses, incluido el mes actual en cero, es decir, vendieron en alguno de los 2 meses anteriores) y al menos `min_months` meses con ventas. Para esos productos la base del "mes actual" es `media_3`:

- Etiquetas de entrenamiento: se agregan las filas con mes actual 0 y `media_3 > 0`; razón = ventas del mes siguiente / `media_3`, con los mismos umbrales (1,15 / 0,90); mes siguiente en cero => Reducir. Las filas de producción conservan su etiqueta y su partición entrenamiento/prueba (las nuevas van a entrenamiento si son anteriores a la primera fila de prueba del producto); solo se agregan filas. Entran también al regresor y al clustering, que usan el mismo marco.
- Predicción: clasificador y regresor como siempre; monto en RF y en la mezcla w*RF + (1-w)*`media_3` con el w de la validación interna de cada mes.
- Clase real en el estudio de caso: ventas reales del mes objetivo contra la misma base (`media_3`) con los mismos umbrales.
- Equivalente más limpio: una sola base de decisión `base = ventas del mes actual si > 0, si no media_3` (función `decision_base`); para producción coincide con la regla de siempre.

Evaluación: mismo estudio de caso de origen rodante (2026-04 a 2026-09, entrenamiento estrictamente anterior a cada mes, ventas reales del holdout; septiembre incompleto). El MASE usa la escala de la Base. Nada se eligió con el holdout. "Previos" son los productos que la Base ya predice; "nuevos", los que agrega la variante.

### Cobertura (millones de COP)

| Mes | Variante | Con decisión | Con monto | Real con monto | Real top 1900 | Cobertura | Real tienda |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-04 | Base | 697 | 645 | 302,4 | 416,0 | 0,727 | 452,9 |
| 2026-04 | Ampliada | 1.080 | 993 | 363,2 | 416,0 | 0,873 | 452,9 |
| 2026-05 | Base | 760 | 709 | 356,5 | 522,4 | 0,682 | 586,3 |
| 2026-05 | Ampliada | 1.121 | 1.015 | 418,7 | 522,4 | 0,802 | 586,3 |
| 2026-06 | Base | 764 | 690 | 230,5 | 306,6 | 0,752 | 337,3 |
| 2026-06 | Ampliada | 1.137 | 1.022 | 263,0 | 306,6 | 0,858 | 337,3 |
| 2026-07 | Base | 724 | 664 | 277,6 | 405,4 | 0,685 | 440,5 |
| 2026-07 | Ampliada | 1.163 | 1.042 | 348,6 | 405,4 | 0,860 | 440,5 |
| 2026-08 | Base | 759 | 678 | 323,8 | 430,7 | 0,752 | 481,2 |
| 2026-08 | Ampliada | 1.165 | 1.023 | 372,1 | 430,7 | 0,864 | 481,2 |
| 2026-09 | Base | 856 | 769 | 301,4 | 427,0 | 0,706 | 462,4 |
| 2026-09 | Ampliada | 1.205 | 1.050 | 354,4 | 427,0 | 0,830 | 462,4 |
| Agrupado | Base | 4.560 | 4.155 | 1.792,1 | 2.507,9 | 0,715 | 2.760,6 |
| Agrupado | Ampliada | 6.871 | 6.145 | 2.119,9 | 2.507,9 | 0,845 | 2.760,6 |

### Total predicho (mezcla) frente a lo que realmente vendieron los productos del top 1900 (millones de COP)

| Mes | Real top 1900 | Base mezcla | Ampliada mezcla | Base RF | Ampliada RF | Base/real | Ampliada/real |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-04 | 416,0 | 247,0 | 280,9 | 163,4 | 157,1 | 0,594 | 0,675 |
| 2026-05 | 522,4 | 269,2 | 308,6 | 151,7 | 153,9 | 0,515 | 0,591 |
| 2026-06 | 306,6 | 327,6 | 369,2 | 184,7 | 176,4 | 1,069 | 1,204 |
| 2026-07 | 405,4 | 259,4 | 310,8 | 127,9 | 127,7 | 0,640 | 0,767 |
| 2026-08 | 430,7 | 266,0 | 309,7 | 132,5 | 138,5 | 0,618 | 0,719 |
| 2026-09 | 427,0 | 256,3 | 284,7 | 120,6 | 121,7 | 0,600 | 0,667 |
| Agrupado | 2.507,9 | 1.625,5 | 1.863,9 | 880,7 | 875,3 | 0,648 | 0,743 |

### Decisiones, agrupado (6 meses)

| Conjunto | n | F1 macro | Acc. balanceada | Accuracy | Mayoritaria (accuracy) | % Reforzar pred. / real |
|---|---:|---:|---:|---:|---:|---:|
| Base, todos | 4.560 | 0,4243 | 0,4575 | 0,5682 | 0,6489 | 54,2 / 29,2 |
| Ampliada, todos | 6.871 | 0,4067 | 0,4478 | 0,5282 | 0,6586 | 55,6 / 29,3 |
| Ampliada, previos | 4.560 | 0,4335 | 0,4706 | 0,5553 | 0,6489 | 50,1 / 29,2 |
| Ampliada, nuevos | 2.311 | 0,3299 | 0,3804 | 0,4747 | 0,6776 | 66,3 / 29,6 |

### Montos, agrupado (productos con monto; MASE con la escala de la Base)

| Conjunto / predicción | n | MAE | RMSE | MASE | WAPE | pred/real | Pred. (M) | Real (M) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base / RF | 4.155 | 314.158 | 1.026.148 | 1,2576 | 0,7284 | 0,4914 | 880,7 | 1.792,1 |
| Base / mezcla | 4.155 | 327.702 | 936.047 | 1,3118 | 0,7598 | 0,9071 | 1.625,5 | 1.792,1 |
| Ampliada, todos / RF | 6.145 | 266.266 | 909.623 | 1,0658 | 0,7718 | 0,4129 | 875,3 | 2.119,9 |
| Ampliada, todos / mezcla | 6.145 | 284.569 | 831.264 | 1,1391 | 0,8249 | 0,8792 | 1.863,9 | 2.119,9 |
| Ampliada, previos / RF | 4.155 | 315.724 | 1.032.982 | 1,2638 | 0,7320 | 0,4817 | 863,3 | 1.792,1 |
| Ampliada, previos / mezcla | 4.155 | 328.257 | 937.557 | 1,3140 | 0,7611 | 0,9043 | 1.620,7 | 1.792,1 |
| Ampliada, nuevos / RF | 1.990 | 163.000 | 571.899 | 0,6525 | 0,9894 | 0,0366 | 12,0 | 327,9 |
| Ampliada, nuevos / mezcla | 1.990 | 193.353 | 546.296 | 0,7740 | 1,1736 | 0,7420 | 243,3 | 327,9 |

### Lectura honesta

- Cobertura y total: la variante cubre 84,5 % de las ventas reales del top 1900 frente a 71,5 % de la Base, y el total predicho por la mezcla pasa de 0,648 a 0,743 de lo que vendieron esos productos. Sigue subestimando.
- Los productos nuevos no se predicen bien en decisión: F1 macro 0,3299 y accuracy 0,4747, por debajo de la clase mayoritaria (0,6776), y recomiendan Reforzar en 66,3 % frente a 29,6 % real. Al añadirlos, el F1 macro del conjunto baja de 0,4243 a 0,4067.
- En monto, el RF casi no aporta para los nuevos (razón de sumas 0,0366): el aporte al total viene de `media_3` dentro de la mezcla (razón 0,7420), es decir, es casi la línea base. El MAE y el MASE de los nuevos son menores porque sus ventas son pequeñas; no son comparables con los de la Base (conjuntos distintos).
- Los productos previos cambian poco (mezcla MAE 328.257 frente a 327.702; F1 macro 0,4335 frente a 0,4243, dentro del ruido de reentrenar con filas adicionales).
- Limitaciones: seis orígenes, septiembre incompleto, un solo valor de la regla (ventana de 3 meses, base `media_3`); no se probaron otras bases ni umbrales. Nada se decidió con el holdout.

### Conclusión y recomendación

La ampliación sí acerca el total a lo que vendieron los productos principales (de 64,8 % a 74,3 %), pero lo hace con la media de 3 meses, no con el modelo, y sus decisiones para los productos nuevos son peores que la clase mayoritaria. Recomendación: no llevar la decisión ampliada a producción. Si se quiere mejorar el total, la opción defendible es incluir esos productos solo en el monto (mezcla, equivalente a `media_3`) y dejar su decisión como "sin recomendación"; esa decisión de producto queda para el usuario y requeriría su propio estudio (por ejemplo, una clase separada o un umbral distinto para estos productos).

## [Datos] V6 hasta el 31 de marzo - 2026-10-13

La V6 se había cortado el 27 de marzo de 2026 solo para reflejar la fecha de exportación de la V5, de modo que marzo quedaba incompleto en el entrenamiento. Eso lo hacía parecer más débil, bajaba la base y la media de 3 meses de la mezcla y, en abril de 2026, el dashboard predijo 198,6 M COP frente a 293,4 M reales para los mismos 606 productos. Ahora la V6 llega hasta el 31 de marzo y el holdout empieza el 1 de abril (el periodo sigue siendo los últimos dos años, 2024-01 a 2026-03). Mismos filtros (facturas tipos 1 y 13, no anuladas, sin servicios), misma consulta (`sql/extraccion_ventas_v6.sql`) y mismo extractor, sobre la misma copia de la base; solo cambian `--desde/--hasta`.

Verificación de la extracción: ningún día aparece en ambos archivos y la unión (V6 + holdout) es idéntica fila a fila a la anterior (61.374 filas), es decir, solo se movieron al entrenamiento las filas del 28 al 31 de marzo.

### Dataset

| | Antes (hasta 27/03) | Después (hasta 31/03) |
|---|---:|---:|
| V6: filas (producto-día) | 47.749 | 48.028 |
| V6: productos | 3.573 | 3.584 |
| V6: monto (COP) | 6.344.817.681 | 6.445.554.996 |
| V6: rango | 2024-01-09 a 2026-03-27 | 2024-01-09 a 2026-03-31 |
| Ventas de marzo de 2026 en el entrenamiento (COP) | 315.329.441 | 416.066.756 |
| Holdout: filas | 13.625 | 13.346 |
| Holdout: productos | 2.462 | 2.439 |
| Holdout: monto (COP) | 2.861.381.559 | 2.760.644.244 |
| Holdout: rango | 2026-03-28 a 2026-09-22 | 2026-04-01 a 2026-09-22 |

### Métricas de la prueba interna (n = 6.242 en el clasificador; n = 4.472 en el regresor)

| Clasificador | Antes | Después |
|---|---:|---:|
| F1 macro | 0,4467 | 0,4501 |
| Accuracy balanceada | 0,5030 | 0,5043 |
| % Reforzar real | 16,63 % | 16,98 % |
| % Reforzar predicho | 28,15 % | 28,15 % |

| Regresor | RF antes | RF después | Mezcla antes | Mezcla después |
|---|---:|---:|---:|---:|
| MAE | 227.468 | 232.349 | 252.409 | 266.174 |
| RMSE | 896.837 | 907.724 | 841.440 | 860.530 |
| R² | 0,3724 | 0,4213 | 0,4475 | 0,4799 |
| MASE | 1,0164 | 1,0382 | 1,1278 | 1,1893 |
| Razón de sumas (pred./real) | 0,6266 | 0,5937 | 0,9950 | 1,0126 |
| Peso w | | | 0,5 | 0,4 |

k = 3 sigue siendo el mejor por silueta (0,3919; k = 2: 0,3607, k = 4: 0,3445). El ajuste de hiperparámetros vuelve a proponer `max_depth` sin límite, con F1 macro de prueba 0,4502 frente a 0,4501 de los valores por defecto: diferencia sin relevancia, así que `DEFAULT_CLF_PARAMS` no cambia (decisión pendiente del usuario).

### Estudio de caso (agrupado, abril a septiembre de 2026)

No cambia: cada mes se entrena con la V6 más el holdout anterior a ese mes, y la unión de datos es la misma, por lo que `estudio_caso.txt`, `estudio_caso.csv` y el experimento de sensibilidad dan los mismos números (F1 macro agrupado 0,4243; MAE modelo 314.158 y mezcla 327.702; razón de sumas 0,4914 y 0,9071). Lo mismo vale para `experimento_mejoras.txt`.

### Comparación de abril de 2026 en el dashboard (`scripts/entrenar.py`)

| | Antes | Después |
|---|---:|---:|
| Productos con predicción de monto | 606 | 645 |
| Suma `ventas_predichas` (mezcla) | 198,6 M | 247,0 M |
| Suma `ventas_predichas_rf` | 129,8 M | 163,4 M |
| Peso w | 0,5 | 0,4 |
| Ventas reales de abril de esos mismos productos | 293,4 M | 302,4 M |
| Mezcla / real | 0,677 | 0,817 |
| Ventas reales de abril, top 1900 | 416,0 M | 416,0 M |
| Ventas reales de abril, todo el taller | 452,9 M | 452,9 M |
| Productos con predicción (total) | 654 | 697 |
| % Reforzar en la predicción de abril | 69,0 % | 66,7 % |

Se reejecutaron `eda.py`, `select_k.py`, `tune_rf.py`, `evaluate_model.py`, `compare_models.py`, `estudio_caso.py`, `experimento_mejoras.py`, `experimento_sensibilidad_gap.py` y `entrenar.py`. En `eda_resumen.txt` marzo de 2026 ya figura como mes completo (promedio de marzo en el resumen estacional del EDA: 151,7 M a 185,3 M). En `comparacion_modelos.txt` el F1 macro del clasificador pasa de 0,4467 a 0,4501.

## [Experimento] Sensibilidad al tramo de baja facturación - 2026-10-13

Experimento sin cambios en el comportamiento de producción: `SalesForecastModel` gana el atributo opcional `sample_weight_fn` (por defecto `None`: no se pasa `sample_weight`, el ajuste es el de siempre); los datos oficiales (`Query_Result_V6.csv`) y el holdout no se tocan. Entre 2024-02 y 2025-04 las facturas de venta caen de ~1.500 a ~450 por mes (la cuenta 4135.06 cae igual) y se recuperan en 2025-05. Pregunta: ¿tratar ese tramo mejora la predicción en meses reales no vistos? Mismo estudio de caso de origen rodante (2026-04 a 2026-09, entrenamiento estrictamente anterior a cada mes), evaluado SIEMPRE contra las ventas reales del holdout. Informe: `outputs/reports/experimento_sensibilidad_gap.txt/.csv`; figura `outputs/figures/experimento_sensibilidad_gap.png`.

Variantes: **Base** (datos oficiales); **V1, datos ajustados (imputación experimental)**: cada producto reescala `VENTAS` y `TOTAL_VENDIDO` de sus filas del tramo por el cociente entre sus ventas medias mensuales de 2025-05 a 2025-10 y las del tramo (meses sin venta cuentan cero), acotado a [1, 5]; con menos de 3 meses con ventas en una ventana se usa el factor global; `PRECIO_PROMEDIO` y `FRECUENCIA` no se modifican; no se crean filas; el CSV ajustado solo existió en un directorio temporal; **V2**, sin datos nuevos: peso de muestra 0,3 (y 0 = excluir) a las filas cuyo mes actual o mes objetivo cae en el tramo.

Factores de V1: 3.574 productos (806 con factor propio, 2.768 con el global); factor global 5,000 (tope); mediana 5,000, mínimo 1,000, máximo 5,000 (el tope se alcanza en la mayoría); 49,5 % de las filas cae en el tramo y 48,8 % se reescala; monto del tramo 1.224,7 M -> 5.277,4 M COP (81,6 -> 351,8 M/mes; referencia 2025-05 a 2025-10: 472,1 M/mes).

### Resultados agrupados (6 meses; MASE con la escala de la Base)

| Variante | F1 macro | Acc. balanceada | Accuracy | % Reforzar pred. / real |
|---|---:|---:|---:|---:|
| Base (datos oficiales) | 0,4243 | 0,4575 | 0,5682 | 54,2 / 29,2 |
| V1: datos ajustados (imputación experimental) | 0,4539 | 0,4742 | 0,6105 | 46,7 / 29,2 |
| V2: peso 0,3 al tramo | 0,4284 | 0,4610 | 0,5746 | 53,3 / 29,2 |
| V2: tramo excluido (peso 0) | 0,4325 | 0,4595 | 0,5816 | 50,9 / 29,2 |
| Clase mayoritaria | 0,2624 | | 0,6489 | |

| Variante / predicción | MAE | RMSE | MASE | WAPE | pred/real |
|---|---:|---:|---:|---:|---:|
| Base / RF solo | 314.158 | 1.026.148 | 1,2576 | 0,7284 | 0,4914 |
| Base / mezcla | 327.702 | 936.047 | 1,3118 | 0,7598 | 0,9071 |
| V1 / RF solo | 329.614 | 1.062.892 | 1,3198 | 0,7516 | 0,3449 |
| V1 / mezcla | 333.687 | 945.854 | 1,3361 | 0,7609 | 0,8752 |
| V2 peso 0,3 / RF solo | 317.659 | 1.023.804 | 1,2716 | 0,7365 | 0,5173 |
| V2 peso 0,3 / mezcla | 329.134 | 937.464 | 1,3175 | 0,7631 | 0,9153 |
| V2 peso 0 / RF solo | 324.655 | 1.025.029 | 1,2996 | 0,7527 | 0,5651 |
| V2 peso 0 / mezcla | 329.855 | 938.818 | 1,3204 | 0,7648 | 0,9243 |

En abril de 2026 el sistema recomienda Reforzar para 66,7 % de los productos (Base) frente a 32,4 % real; V1 52,7 %, V2 0,3 65,3 %, V2 0 62,4 %.

### Sumas mensuales en millones de COP (productos con predicción de monto)

| Mes | Real | Base RF | Base mezcla | V1 RF | V1 mezcla | V2 0,3 RF | V2 0,3 mezcla | V2 0 RF | V2 0 mezcla |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-04 | 302,4 | 163,4 | 247,0 | 89,7 | 217,5 | 177,5 | 252,7 | 200,1 | 251,4 |
| 2026-05 | 356,5 | 151,7 | 269,2 | 105,1 | 255,1 | 164,1 | 272,9 | 197,3 | 282,9 |
| 2026-06 | 230,5 | 184,7 | 327,6 | 107,1 | 312,0 | 190,3 | 328,7 | 206,1 | 331,9 |
| 2026-07 | 277,6 | 127,9 | 259,4 | 105,6 | 273,5 | 136,7 | 262,0 | 141,8 | 263,6 |
| 2026-08 | 323,8 | 132,5 | 266,0 | 107,5 | 258,4 | 142,7 | 269,1 | 148,7 | 270,8 |
| 2026-09 | 301,4 | 120,6 | 256,3 | 102,7 | 250,9 | 115,8 | 254,9 | 118,8 | 255,8 |
| Agrupado | 1.792,1 | 880,7 | 1.625,5 | 617,8 | 1.567,5 | 927,0 | 1.640,3 | 1.012,8 | 1.656,4 |

### Etiquetas de entrenamiento en los límites del tramo (modelo de 2026-04; [Reducir, Mantener, Reforzar])

| Variante | 2024-01 -> 2024-02 (caída) | 2025-04 -> 2025-05 (recuperación) |
|---|---|---|
| Base, V2 0,3 y V2 0 (mismas etiquetas) | [660, 15, 51] | [469, 36, 287] |
| V1: datos ajustados (imputación experimental) | [527, 28, 171] | [651, 25, 116] |

Mecanismo: con los datos oficiales la recuperación de mayo de 2025 aparece como 287 "Reforzar" de 792 filas y la caída de febrero de 2024 como 660 "Reducir"; V1 aplana ambos saltos (116 "Reforzar" en la recuperación).

### Conclusión

Con datos reales de seis meses, ninguna variante mejora de forma consistente. V1 sube el F1 macro agrupado (0,4243 -> 0,4539) y la accuracy balanceada (0,4575 -> 0,4742) y baja el porcentaje de Reforzar (54,2 % -> 46,7 %, aún lejos del 29,2 % real), pero el efecto no es uniforme (en 2026-05 el F1 macro baja de 0,4487 a 0,4267) y empeora los montos: MAE del RF 314.158 -> 329.614, razón de suma del RF 0,4914 -> 0,3449 y mezcla 0,9071 -> 0,8752. V2 casi no cambia la clasificación (F1 macro +0,004 con peso 0,3, +0,008 con peso 0) y deja los montos igual o algo peores. Ninguna variante supera a la clase mayoritaria en accuracy (0,6489). Limitaciones: seis meses de un solo holdout, la ganancia de V1 es de unos 0,03 de F1 macro sin intervalo de confianza, el tope ×5 se satura en la mayoría de los productos y no se sabe si la caída es una caída real de ventas o ventas fuera del sistema. Recomendación: mantener los datos oficiales. Si se quisiera explorar V1, sería solo como recomendación para la decisión de reposición (no para los montos) y tras confirmar con el taller el origen de la caída; no se adopta ni se reporta como resultado del sistema.

## [Sistema] Predicción combinada en el dashboard - 2026-10-13

**Qué cambia.** La predicción de ventas en pesos del sistema pasa a ser la mezcla `w x RF + (1 - w) x media_3` (opción 3b del experimento "[Experimento] Mejoras de la predicción en pesos"), donde `media_3` es el promedio de los últimos tres meses del producto (`media_3_actual`, con respaldo a las ventas del mes actual si falta) y `w` toma valores de 0, 0,1 ... 1.

**Por qué.** El regresor en `log1p` subestima el total en pesos (en la prueba interna predice 0,63 de la suma real); en el estudio de caso fuera de muestra la mezcla subió la razón suma predicha/real de 0,49 a 0,91 con un aumento de MAE de 4,3 % (314.158 -> 327.702).

**Cómo se elige w.** Solo con la partición de prueba interna del regresor dentro del entrenamiento (`train_regressor`), minimizando |suma predicha - suma real|; no se usa ningún mes objetivo ni holdout. Clasificador, k, hiperparámetros, umbrales y valores por defecto no cambian.

**Cambios en el código.**
- `SalesForecastModel`: nuevos `blend_weight`, `blend_metrics` (MAE, MSE, RMSE, R², MASE, razón de sumas, w) y `reg_test_media_3`; `reg_metrics` recibe además `sum_ratio`. `predict_next_month` entrega `ventas_predichas` (mezcla) y `ventas_predichas_rf` (RF puro); los productos sin predicción de monto siguen sin ella.
- Dashboard: `/api/summary` devuelve `pred_total` (mezcla), `pred_total_rf` y `blend_weight`; el KPI "Ventas predichas" muestra la mezcla y la nota "Combinación del modelo (w %) y el promedio de los últimos 3 meses"; la leyenda del historial dice "Predicción próximo mes (modelo combinado)"; el detalle del producto muestra "Solo modelo: $..."; `/api/products` y el CSV descargable incluyen ambas columnas.
- `scripts/evaluate_model.py` agrega la sección "Mezcla (modelo + media de 3 meses)" junto a la fila del RF; `scripts/estudio_caso.py` reporta "modelo" (RF) y "mezcla"; `scripts/experimento_mejoras.py` usa `ventas_predichas_rf` como base (sus números no cambian).
- Pruebas: `tests/test_blend_model.py` y `tests/test_dashboard_blend.py`; suite completa 142 pruebas OK.

### Corrida real (V6 sin servicios, `scripts/entrenar.py` + `scripts/evaluate_model.py`, prueba interna n = 4.472, top 1900 productos)

w elegido: **0,5**.

| Métrica | RF | Mezcla | Media_3 |
|---|---:|---:|---:|
| MAE (COP) | 227.468 | 252.409 | 310.225 |
| RMSE (COP) | 896.837 | 841.440 | 915.553 |
| R² | 0,3724 | 0,4475 | 0,3459 |
| MASE | 1,0164 | 1,1278 | 1,3862 |
| Razón de sumas pred/real | 0,6266 | 0,9950 | 1,3634 |

La mezcla reduce RMSE y sube R² frente al RF, a cambio de un MAE 11,0 % mayor, y deja la suma predicha casi igual a la real en la prueba. Estudio de caso fuera de muestra (agrupado, 6 meses): modelo MAE 314.158 / razón 0,4914; mezcla MAE 327.702 / razón 0,9071 (w por mes entre 0,2 y 0,4).

### Prueba de humo del dashboard (corrida real, abril 2026)

`/api/run` con los valores por defecto sobre `Query_Result_V6.csv`: 654 productos predichos, 606 con monto. `pred_total` = 198.601.648 COP (mezcla), `pred_total_rf` = 129.816.494 COP, `blend_weight` = 0,5. Producto 1888: `ventas_predichas` 18.571 COP, `ventas_predichas_rf` 11.908 COP.

## [Experimento] Mejoras de la predicción en pesos - 2026-10-13

Experimento sin cambios en el modelo de producción (`model.py`, dashboard y `entrenar.py` intactos). Mide en el estudio de caso fuera de muestra (mismos meses 2026-04 a 2026-09, mismas filas y valores reales; `scripts/experimento_mejoras.py`, ayudantes puros en `src/ventas_forecast/improvements.py`, 15 pruebas nuevas) si cuatro opciones acercan la predicción en pesos a la realidad, sobre todo la suma mensual. La base reproduce `estudio_caso.csv` (MAE agrupado 314.158). El clasificador no cambia. **Nada se ajustó ni se eligió con el mes objetivo (holdout)**: los hiperparámetros de la opción 2 son fijos (HistGB Poisson: 300 iteraciones, tasa 0,05, 31 hojas, mínimo 20 por hoja; RF sin log: 200 árboles, profundidad 12) y el peso w de la opción 3 y el cuantil conformal de la opción 4 salen de la partición de prueba interna del entrenamiento de cada mes.

Opciones: 1 reconciliación de arriba hacia abajo (total del mes = suma de la media_3 de los productos con predicción, repartido en proporción al RF; 1b: total = media de los últimos 3 meses de la serie agregada, idéntica a la 1 con diferencia de 0 COP en los 6 meses); 2a HistGB con pérdida Poisson y 2b RF sobre pesos crudos; 3 mezcla w·RF + (1−w)·media_3 con w elegido por MAE (3) o por el sesgo de la suma (3b) en la validación interna; 4 intervalos de 80 % del RF (4a cuantiles 10-90 de los árboles en escala log; 4b conformal dividido con los residuos de la validación interna en escala log).

### Agrupado (6 meses, productos con predicción de monto)

| Variante | MAE | RMSE | MASE | WAPE | Razón suma pred/real | Pred. (M COP) | Real (M COP) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base: RF (log1p) | 314.158 | 1.026.148 | 1,2576 | 0,7284 | 0,4914 | 880,7 | 1.792,1 |
| Media de 3 meses | 353.809 | 961.301 | 1,4163 | 0,8203 | 1,0820 | 1.939,1 | 1.792,1 |
| Op. 1 Arriba-abajo (total media_3) | 421.761 | 1.356.077 | 1,6883 | 0,9779 | 1,0820 | 1.939,1 | 1.792,1 |
| Op. 1b Arriba-abajo (total serie agregada) | 421.761 | 1.356.077 | 1,6883 | 0,9779 | 1,0820 | 1.939,1 | 1.792,1 |
| Op. 2a HistGB Poisson (pesos) | 490.089 | 1.314.335 | 1,9618 | 1,1363 | 1,5928 | 2.854,4 | 1.792,1 |
| Op. 2b RF sin log (pesos) | 540.001 | 1.371.731 | 2,1616 | 1,2520 | 1,7555 | 3.145,9 | 1.792,1 |
| Op. 3 Mezcla, w por MAE | 314.158 | 1.026.148 | 1,2576 | 0,7284 | 0,4914 | 880,7 | 1.792,1 |
| Op. 3b Mezcla, w por sesgo de la suma | 327.702 | 936.047 | 1,3118 | 0,7598 | 0,9071 | 1.625,5 | 1.792,1 |

Pesos elegidos con la validación interna: opción 3 (MAE) w = 1,0 en los 6 meses (siempre el RF puro); opción 3b (sesgo) w = 0,4 / 0,3 / 0,2 / 0,3 / 0,3 / 0,3 para abril a septiembre.

### Suma mensual predicha vs real (millones de COP)

| Mes | Real | Base RF | Media_3 | Op. 1 | Op. 2a Poisson | Op. 2b RF sin log | Op. 3b |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2026-04 | 302,4 | 163,4 | 302,8 | 302,8 | 522,4 | 539,7 | 247,0 |
| 2026-05 | 356,5 | 151,7 | 319,5 | 319,5 | 520,5 | 602,5 | 269,2 |
| 2026-06 | 230,5 | 184,7 | 363,4 | 363,4 | 712,5 | 958,8 | 327,6 |
| 2026-07 | 277,6 | 127,9 | 315,7 | 315,7 | 321,2 | 335,0 | 259,4 |
| 2026-08 | 323,8 | 132,5 | 323,2 | 323,2 | 400,0 | 364,3 | 266,0 |
| 2026-09 | 301,4 | 120,6 | 314,5 | 314,5 | 377,8 | 345,6 | 256,3 |
| Agrupado | 1.792,1 | 880,7 | 1.939,1 | 1.939,1 | 2.854,4 | 3.145,9 | 1.625,5 |

Las ventas reales de septiembre están incompletas (el holdout llega al 22).

### Intervalos de 80 % del RF (opción 4)

| Variante | Cobertura de productos (agrupado) | Ancho medio (COP) | Cobertura por mes |
|---|---:|---:|---|
| 4a Cuantiles 10-90 de los árboles | 61,4 % | 911.609 | 65,0 / 63,8 / 67,5 / 59,6 / 55,2 / 57,7 % |
| 4b Conformal 80 % (escala log) | 81,5 % | 366.290.907 | 81,7 / 80,8 / 80,0 / 82,2 / 85,3 / 79,3 % |

Total mensual: la suma de los límites inferiores y la de los superiores es una cota burda (suma de cuantiles, no cuantil de la suma). Con 4a el total real cae dentro de la cota en los 6 meses (p. ej. abril 78,0 a 786,8 M frente a 302,4 M real), pero la cota es demasiado ancha para servir; con 4b el límite superior suma 151.178,8 a 349.610,6 M por mes.

### Conclusión

- Solo la media_3 y la arriba-abajo dejan el total cerca de la realidad (razón 1,0820), pero la arriba-abajo empeora el error por producto (MAE 421.761 frente a 314.158 de la base y 353.809 de la media_3): el RF reparte mal el total, así que no sirve para decisiones por producto. La opción 1b es idéntica a la 1.
- La mezcla elegida por MAE vuelve siempre al RF puro (w = 1,0): no corrige el total. La mezcla elegida por el sesgo de la suma (3b) es el mejor compromiso entre las opciones evaluadas: razón 0,9071 (suma 1.625,5 M frente a 1.792,1 M reales, contra 880,7 M de la base), MAE 327.702 (+4,3 % sobre la base) y el mejor RMSE agrupado (936.047). Aun así subestima el total en 5 de los 6 meses y lo sobrestima en junio (327,6 frente a 230,5 M), y los pesos salen de la validación interna, que es parte del mismo entrenamiento.
- Las pérdidas sobre pesos crudos (2a, 2b) sobrestiman el total (razones 1,5928 y 1,7555) y empeoran MAE, RMSE y MASE: no ayudan.
- Los intervalos de los árboles (4a) cubren 61,4 % frente a 80 % nominal; los conformales (4b) cubren 81,5 %, pero su ancho medio (366 M COP) los vuelve inútiles en escala de producto.
- Una sola ventana de 6 meses, un holdout con septiembre incompleto y dependencia de una media_3 que acierta el total porque los meses fueron estables: las conclusiones son orientativas, no una validación del cambio de modelo. No se modificó el modelo de producción.

Salidas (no versionadas): `outputs/reports/experimento_mejoras.txt` y `.csv`, `outputs/figures/experimento_mejoras.png`.

## [Datos] V6 sin servicios - 2026-10-13

### Decisión y regla

Decisión del usuario (2026-10-13): la tesis trata de la demanda de repuestos e inventario, y un servicio (mano de obra, flete) no tiene existencias que reforzar ni reducir. En la V6 el "producto" de mayor monto era MANTENIMIENTO (381,6 M COP, mano de obra) y también aparecía SERVICIO DE TORNO.

**Regla elegida: lista explícita de códigos de producto** en `sql/extraccion_ventas_v6.sql` (FILTRO 5, `p.CODIGO NOT IN (...)`), cubierta por `tests/test_extraer_ventas.py::TestConsultaV6`. Códigos: `1250` MANTENIMIENTO, `2660` SERVICIO DE TORNO, `TORNO SPOOL`, `3087` y `FLETE` (ambos TRANSPORTE FLETE) y `SEPILLADADECAMPANAS`. Los códigos 3087, FLETE y SEPILLADADECAMPANAS no tienen ventas en facturas dentro de la extracción (solo ajustes de inventario o compras); se incluyen para que la regla siga siendo válida si se facturan.

Por qué no una bandera de la base (evidencia sobre la copia, ventas de facturas 1 y 13 entre 2024-01-01 y 2026-09-30 con los filtros de la V6):

| Candidato | Resultado | Veredicto |
|---|---|---|
| `PRODUCTO.SERVICIO = '1'` | marca 3 productos vendidos (3,9 M COP): INFLALLANTAS (manguera), un retén de rueda y un tornillo; los tres son repuestos. **No** marca MANTENIMIENTO ni SERVICIO DE TORNO | inconsistente, descartada |
| `PRODUCTO.ESFLETE = '1'` | marca 1 producto vendido (16.000 COP): un acople de manguera | descartada |
| `NOAFECTADOCUMENTO`, `GRATIS`, `PRODUCTOUNICO`, `BANDERAS`, `TIPOMED` | un solo valor para todos los vendidos | no discriminan |
| Grupo VARIOS (`IDGRUPO` 52) | 24 productos vendidos (856 M COP en total): mezcla servicios con repuestos (SOLUCION DE LIMPIEZA INDUSTRIAL 176,1 M, REPUESTOS VARIOS, minifusibles, piezas "SPOOL") | descartado |
| Reglas por nombre (MANTENIMIENTO, SERVICIO, TORNO, SOLDADURA, REPARACION...) | falsos positivos: "UNIDAD DE MANTENIMIENTO DE VASO" (repuesto), "TORN ..." (tornillo), soldaduras en varilla, empaquetaduras de "reparación" | descartada |

**Excluidos (monto en la ventana de la V6, 2024-01-01 a 2026-03-27, y en el holdout, 2026-03-28 a 2026-09-30):**

| Código | Producto | V6 (COP) | Filas V6 | Holdout (COP) |
|---|---|---:|---:|---:|
| 1250 | MANTENIMIENTO | 381.573.192,5 | 671 | 153.561.820 |
| 2660 | SERVICIO DE TORNO | 14.957.050 | 90 | 14.773.600 |
| TORNO SPOOL | TORNO SPOOL | 360.000 | 7 | 10.000 |
| 3087, FLETE, SEPILLADADECAMPANAS | transporte y sepillada | 0 | 0 | 0 |
| **Total** | | **396.890.242,5 (5,89 % de la V6 con servicios)** | **768** | **168.345.420 (5,56 % del holdout)** |

**Conservados por ser productos:** SOLUCION DE LIMPIEZA INDUSTRIAL (176,1 M), SOLDADURA y soldaduras en varilla.
**Ambiguos, no excluidos (pendiente del usuario):** REPUESTOS VARIOS (78,7 M, 2024-01 a 2026-09; ítem comodín de repuestos sin código propio), "VARIOS" (N/A) y las piezas "... SPOOL" (BANDA, CAMARA, BLOQUE, etc.; ~11 M en conjunto, parecen trabajos sobre pieza del cliente), COMPRESOR REPARADO y DISPONIBLE (sin ventas en facturas).

Los CSV se regeneraron con `sql/extraer_ventas.py` sobre la copia de la base (la original no se tocó). Los reportes "antes" se respaldaron en `odd/v6_con_servicios_reports/` (sin versionar).

### Dataset

| | V6 con servicios | V6 sin servicios | Diferencia |
|---|---:|---:|---:|
| Entrenamiento: filas (producto-día) | 48.517 | 47.749 | -768 |
| Entrenamiento: productos | 3.576 | 3.573 | -3 |
| Entrenamiento: monto (COP) | 6.741.707.924 | 6.344.817.681 | -396.890.243 (-5,89 %) |
| Holdout: filas | 13.807 | 13.625 | -182 |
| Holdout: productos | 2.465 | 2.462 | -3 |
| Holdout: monto (COP) | 3.029.726.979 | 2.861.381.559 | -168.345.420 (-5,56 %) |

### Antes (V6 con servicios) y después (V6 sin servicios) del pipeline completo

Números copiados de los reportes (`outputs/reports/`). La razón suma predicha/real de la prueba interna no está en los reportes: se calculó con `reg_test_results` de `train_regressor()` (0,6717 con servicios).

**EDA**

| | Con servicios | Sin servicios |
|---|---:|---:|
| Filas producto-mes tras calendario completo | 66.236 | 66.155 |
| Productos tras limpieza | 3.576 | 3.573 |
| Producto-mes con venta cero | 65,53 % | 65,57 % |
| Mediana de meses en cero por producto (Q1 / Q3) | 75,00 % (51,85 / 87,50) | 75,00 % (51,85 / 87,50) |
| Pareto A / B / C (productos) | 385 (10,8 %) / 661 (18,5 %) / 2.530 (70,7 %) | 408 (11,4 %) / 667 (18,7 %) / 2.498 (69,9 %) |
| Ventas de A / B / C | 79,99 / 15,01 / 5,00 % | 79,98 / 15,01 / 5,00 % |
| Promedio de ventas de julio (máximo) / abril (mínimo), COP | 436.906.414 / 119.701.508 | 419.462.940 / 106.155.984 |

**Selección de k (silueta)**

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Con servicios | 0,3598 | 0,3940 | 0,3427 | 0,3572 | 0,3588 | 0,3687 | 0,3433 | 0,3488 | 0,3440 |
| Sin servicios | 0,3607 | 0,3919 | 0,3445 | 0,3555 | 0,3581 | 0,3694 | 0,3452 | 0,3456 | 0,3455 |

k = 3 sigue teniendo la mayor silueta (0,3919).

**Ajuste de hiperparámetros del clasificador**

| | Con servicios | Sin servicios |
|---|---|---|
| Mejores parámetros | `max_depth` 14, `min_samples_leaf` 3, `n_estimators` 300 | `max_depth` None, `min_samples_leaf` 3, `n_estimators` 300 |
| F1 macro en validación cruzada | 0,4366 ± 0,0131 | 0,4377 ± 0,0086 |
| F1 macro de prueba, por defecto / ajustado | 0,4507 / 0,4507 | 0,4467 / 0,4475 |
| Acc. balanceada de prueba, por defecto / ajustado | 0,5166 / 0,5166 | 0,5030 / 0,4971 |

El mejor ajuste difiere de `DEFAULT_CLF_PARAMS` solo en `max_depth` (None frente a 14); en prueba la diferencia es 0,0008 de F1 macro y la accuracy balanceada empeora, así que no es una mejora clara. Se conservan los valores por defecto; la decisión queda pendiente del usuario.

**Clasificador (prueba)**

| | Con servicios | Sin servicios |
|---|---:|---:|
| n prueba | 6.255 | 6.242 |
| Distribución Reducir / Mantener / Reforzar | 80,18 / 3,13 / 16,69 % | 80,25 / 3,12 / 16,63 % |
| F1 macro | 0,4507 | 0,4467 |
| Accuracy balanceada | 0,5166 | 0,5030 |
| Accuracy | 0,6614 | 0,6668 |
| F1 ponderado | 0,7089 | 0,7123 |
| F1 Reducir / Mantener / Reforzar | 0,78 / 0,10 / 0,47 | 0,79 / 0,08 / 0,47 |
| Línea base mayoritaria: F1 macro / acc. balanceada / accuracy / F1 ponderado | 0,2967 / 0,3333 / 0,8018 / 0,7135 | 0,2968 / 0,3333 / 0,8025 / 0,7145 |

Matriz de confusión sin servicios (filas = real; orden Reducir, Mantener, Reforzar): [3465, 532, 1012], [77, 35, 83], [262, 114, 662].

**Regresor (prueba, top productos)**

| | Con servicios | Sin servicios |
|---|---:|---:|
| n prueba | 4.482 | 4.472 |
| MAE modelo / ingenuo / media_3 | 244.042 / 431.137 / 315.215 | 227.468 / 424.427 / 310.225 |
| RMSE modelo / ingenuo / media_3 | 973.944 / 1.273.318 / 929.992 | 896.837 / 1.258.104 / 915.553 |
| R² modelo / ingenuo / media_3 | 0,4744 / 0,1016 / 0,5208 | 0,3724 / -0,2351 / 0,3459 |
| MASE modelo / ingenuo / media_3 | 1,0748 / 1,8988 / 1,3883 | 1,0164 / 1,8964 / 1,3862 |
| Razón suma predicha / suma real | 0,6717 | 0,6266 |

Sin servicios, el modelo supera a media_3 también en RMSE y R² (con servicios solo lo superaba en MAE y MASE), y el MASE baja a 1,0164 (sigue sobre 1).

**Comparación de modelos**

| Clasificador | RF con / sin | XGBoost con / sin | Mayoritaria con / sin |
|---|---:|---:|---:|
| F1 macro | 0,4507 / 0,4467 | 0,4350 / 0,4352 | 0,2967 / 0,2968 |
| Acc. balanceada | 0,5166 / 0,5030 | 0,5190 / 0,5157 | 0,3333 / 0,3333 |
| Accuracy | 0,6614 / 0,6668 | 0,6096 / 0,6097 | 0,8018 / 0,8025 |
| F1 ponderado | 0,7089 / 0,7123 | 0,6790 / 0,6791 | 0,7135 / 0,7145 |

| Regresión A (con / sin) | RF | XGBoost | Ingenuo | Media_3 |
|---|---:|---:|---:|---:|
| MAE | 244.042 / 227.468 | 258.181 / 248.134 | 431.137 / 424.427 | 315.215 / 310.225 |
| RMSE | 973.944 / 896.837 | 1.041.222 / 952.489 | 1.273.318 / 1.258.104 | 929.992 / 915.553 |
| R² | 0,4744 / 0,3724 | 0,3993 / 0,2921 | 0,1016 / -0,2351 | 0,5208 / 0,3459 |
| MASE | 1,0748 / 1,0164 | 1,1371 / 1,1087 | 1,8988 / 1,8964 | 1,3883 / 1,3862 |

| Regresión B (n 339 con / 332 sin) | RF | XGBoost | Ingenuo | Media_3 | ARIMA | Prophet |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 146.781 / 121.963 | 154.674 / 128.974 | 311.407 / 282.368 | 234.136 / 241.987 | 214.765 / 171.728 | 191.777 / 194.072 |
| MASE | 0,6465 / 0,5450 | 0,6812 / 0,5763 | 1,3715 / 1,2617 | 1,0312 / 1,0813 | 0,9459 / 0,7673 | 0,8446 / 0,8672 |

**Estudio de caso (origen rodante, abril a septiembre)**

Decisiones, con servicios -> sin servicios (entre paréntesis, la línea base mayoritaria sin servicios):

| Mes | n | F1 macro | Acc. balanceada | Accuracy |
|---|---:|---:|---:|---:|
| 2026-04 | 699 -> 697 | 0,3785 -> 0,3599 (0,2510) | 0,4230 -> 0,4114 | 0,5150 -> 0,5165 (0,6040) |
| 2026-05 | 762 -> 760 | 0,4451 -> 0,4487 (0,2554) | 0,4708 -> 0,4729 | 0,5564 -> 0,5579 (0,6211) |
| 2026-06 | 766 -> 764 | 0,3521 -> 0,3652 (0,2826) | 0,4371 -> 0,4470 | 0,5039 -> 0,5105 (0,7356) |
| 2026-07 | 727 -> 724 | 0,4663 -> 0,4769 (0,2499) | 0,4866 -> 0,4947 | 0,6107 -> 0,6119 (0,5994) |
| 2026-08 | 761 -> 759 | 0,4358 -> 0,4329 (0,2502) | 0,4493 -> 0,4450 | 0,5848 -> 0,5916 (0,6008) |
| 2026-09 | 858 -> 856 | 0,4415 -> 0,4433 (0,2785) | 0,4812 -> 0,4849 | 0,6119 -> 0,6133 (0,7173) |
| Agrupado | 4.573 -> 4.560 | 0,4228 -> 0,4243 (0,2624) | 0,4570 -> 0,4575 | 0,5651 -> 0,5682 (0,6489) |

Matriz de confusión agrupada sin servicios (filas = real; orden Reducir, Mantener, Reforzar): [1554, 118, 1287], [77, 23, 171], [265, 51, 1014].

Montos (productos con predicción de monto), con servicios -> sin servicios:

| Mes | n | MAE modelo | MASE modelo | Razón suma pred./real: modelo / ingenuo / media_3 |
|---|---:|---:|---:|---|
| 2026-04 | 647 -> 645 | 359.108 -> 327.119 | 1,5816 -> 1,4616 | 0,5927 -> 0,5403 / 1,1565 -> 1,1809 / 0,9805 -> 1,0014 |
| 2026-05 | 711 -> 709 | 357.725 -> 358.710 | 1,4936 -> 1,5213 | 0,4921 -> 0,4254 / 1,1006 -> 1,0966 / 0,9064 -> 0,8962 |
| 2026-06 | 691 -> 690 | 259.341 -> 243.365 | 1,0477 -> 1,0000 | 0,8662 -> 0,8013 / 2,1010 -> 2,1372 / 1,5864 -> 1,5767 |
| 2026-07 | 666 -> 664 | 324.761 -> 306.074 | 1,2623 -> 1,2101 | 0,5031 -> 0,4608 / 0,9506 -> 0,9985 / 1,1015 -> 1,1374 |
| 2026-08 | 680 -> 678 | 377.686 -> 364.083 | 1,4047 -> 1,3735 | 0,4498 -> 0,4091 / 1,1336 -> 1,1492 / 0,9778 -> 0,9982 |
| 2026-09 | 771 -> 769 | 287.783 -> 288.696 | 1,0357 -> 1,0528 | 0,4567 -> 0,4002 / 1,3302 -> 1,3395 / 1,0416 -> 1,0435 |
| Agrupado | 4.166 -> 4.155 | 326.665 -> 314.158 | 1,2879 -> 1,2576 | 0,5446 -> 0,4914 / 1,2572 -> 1,2798 / 1,0714 -> 1,0820 |

MAE agrupado sin servicios: modelo 314.158, ingenuo 447.253 (antes 459.866), media_3 353.809 (antes 363.233). RMSE agrupado: modelo 1.026.148, ingenuo 1.190.872, media_3 961.301. El MAE del modelo es menor que el del ingenuo y que el de media_3 todos los meses (con servicios, abril era la excepción frente a media_3; sin servicios abril queda 327.119 frente a 327.493).

**Sumas en pesos del estudio de caso (nuevo; millones de COP; las cuatro primeras columnas son sobre los productos con predicción de monto)**

| Mes | Suma predicha (modelo) | Suma real | Suma ingenuo | Suma media_3 | Ventas reales totales del mes |
|---|---:|---:|---:|---:|---:|
| 2026-04 | 163,4 | 302,4 | 357,0 | 302,8 | 452,9 |
| 2026-05 | 151,7 | 356,5 | 391,0 | 319,5 | 586,3 |
| 2026-06 | 184,7 | 230,5 | 492,5 | 363,4 | 337,3 |
| 2026-07 | 127,9 | 277,6 | 277,2 | 315,7 | 440,5 |
| 2026-08 | 132,5 | 323,8 | 372,1 | 323,2 | 481,2 |
| 2026-09 | 120,6 | 301,4 | 403,7 | 314,5 | 462,4 |
| Agrupado | 880,7 | 1.792,1 | 2.293,5 | 1.939,1 | 2.760,6 |

El modelo suma entre 0,40 y 0,80 de lo real en los productos con predicción de monto. Esos productos son entre 61 % y 68 % de las ventas totales de cada mes (agrupado 64,9 %), de modo que la suma predicha equivale a 0,32 de la venta total (880,7 / 2.760,6). El estudio de caso anterior no guardó las sumas en pesos, por lo que no hay columna "antes" para esta tabla (las razones sí están arriba). Las ventas de septiembre llegan hasta el 22.

**Lectura general**: excluir los servicios casi no mueve la estructura (k = 3, Pareto, intermitencia). El clasificador queda en 0,4467 de F1 macro (-0,004) y 0,5030 de accuracy balanceada (-0,014), sin cambiar la conclusión frente a la línea base mayoritaria (gana en ambas métricas y pierde en accuracy). El regresor mejora en MAE, RMSE y MASE y ya supera a media_3 en todas las métricas de la prueba interna; el R² baja (0,4744 a 0,3724). El sesgo a la baja del total persiste (0,6266 en prueba; 0,4914 en el estudio de caso).

## [Evaluación] Estudio de caso fuera de muestra (origen rodante) - 2026-10-13

`scripts/estudio_caso.py` (con los ayudantes probados de `src/ventas_forecast/case_study.py` y `tests/test_case_study.py`) evalúa el procedimiento desplegado mes a mes con ventas reales posteriores al entrenamiento (`data/holdout/ventas_2026_04_09.csv`, de 2026-03-28 a 2026-09-22). Para cada mes objetivo M (2026-04 a 2026-09) se entrena `SalesForecastModel` con la V6 más el holdout estrictamente anterior a M, con los valores por defecto del dashboard (crecimiento 1,15; reducción 0,90; k = 3; `min_months` = 6; `top_n_products` = 1900), se predice M como lo hace `entrenar.py` y se compara con las ventas reales de M. La clase real se construye con la regla de las etiquetas de entrenamiento (ventas reales de M / ventas del mes anterior; un mes sin ventas implica Reducir). La línea base siempre predice Reducir (clase mayoritaria del entrenamiento de cada mes). Los meses del holdout nunca se usaron para elegir k ni los hiperparámetros. Salidas: `outputs/reports/estudio_caso.txt` y `.csv`, `outputs/figures/estudio_caso.png`. Las ventas de septiembre están incompletas (el holdout llega al 22).

**Decisiones (F1 macro / accuracy balanceada / accuracy; entre paréntesis la línea base mayoritaria)**

| Mes | n | F1 macro | Acc. balanceada | Accuracy |
|---|---:|---:|---:|---:|
| 2026-04 | 699 | 0,3785 (0,2510) | 0,4230 (0,3333) | 0,5150 (0,6037) |
| 2026-05 | 762 | 0,4451 (0,2553) | 0,4708 (0,3333) | 0,5564 (0,6207) |
| 2026-06 | 766 | 0,3521 (0,2827) | 0,4371 (0,3333) | 0,5039 (0,7363) |
| 2026-07 | 727 | 0,4663 (0,2496) | 0,4866 (0,3333) | 0,6107 (0,5983) |
| 2026-08 | 761 | 0,4358 (0,2498) | 0,4493 (0,3333) | 0,5848 (0,5992) |
| 2026-09 | 858 | 0,4415 (0,2783) | 0,4812 (0,3333) | 0,6119 (0,7168) |
| Agrupado | 4.573 | 0,4228 (0,2622) | 0,4570 (0,3333) | 0,5651 (0,6484) |

Matriz de confusión agrupada (filas = real; orden Reducir, Mantener, Reforzar): [1541, 137, 1287], [79, 24, 170], [253, 63, 1019].

**Montos (productos con predicción de monto; MAE y RMSE en COP)**

| Mes | n | MAE modelo / ingenuo / media_3 | MASE modelo / ingenuo / media_3 | Suma pred./real: modelo / ingenuo / media_3 |
|---|---:|---|---|---|
| 2026-04 | 647 | 359.108 / 485.821 / 345.090 | 1,5816 / 2,1397 / 1,5199 | 0,5927 / 1,1565 / 0,9805 |
| 2026-05 | 711 | 357.725 / 461.459 / 393.584 | 1,4936 / 1,9268 / 1,6434 | 0,4921 / 1,1006 / 0,9064 |
| 2026-06 | 691 | 259.341 / 560.997 / 361.622 | 1,0477 / 2,2662 / 1,4608 | 0,8662 / 2,1010 / 1,5864 |
| 2026-07 | 666 | 324.761 / 397.466 / 368.163 | 1,2623 / 1,5449 / 1,4310 | 0,5031 / 0,9506 / 1,1015 |
| 2026-08 | 680 | 377.686 / 440.465 / 395.473 | 1,4047 / 1,6382 / 1,4709 | 0,4498 / 1,1336 / 0,9778 |
| 2026-09 | 771 | 287.783 / 416.993 / 319.219 | 1,0357 / 1,5007 / 1,1489 | 0,4567 / 1,3302 / 1,0416 |
| Agrupado | 4.166 | 326.665 / 459.866 / 363.233 | 1,2879 / 1,8131 / 1,4321 | 0,5446 / 1,2572 / 1,0714 |

RMSE agrupado: modelo 1.054.010, ingenuo 1.233.164, media_3 990.914.

**Lectura**: las decisiones superan a la línea base mayoritaria en F1 macro y en accuracy balanceada todos los meses (pierden en accuracy en cinco de seis, como en la prueba interna); el F1 macro agrupado (0,4228) es algo menor que el de la prueba interna (0,4507). El MAE del modelo es menor que el del ingenuo todos los meses y menor que el de media_3 en cinco de seis (abril es la excepción); la media_3 tiene menor RMSE en el agrupado. El MASE queda sobre 1 todos los meses. La suma predicha es entre 0,45 y 0,87 de la real (agrupada 0,5446): el sesgo a la baja del regresor entrenado en log1p persiste fuera de muestra, y la media_3 y el ingenuo no lo tienen.

## [Datos] V6: solo facturas de venta - 2026-10-13

### Problema

La consulta V5 (`sql/extraccion_ventas_v5.sql`) no filtraba el tipo de documento (`DOCINV.IDTIPODOCINV`) ni los anulados. Sobre la base del taller, en el periodo de la V5 solo ~51 % del monto (`TOTAL_VENDIDO`) eran facturas de venta (tipos 1 y 13); el resto eran ajustes de inventario (tipos 10 y 11, ~35 %), pedidos (5), compras (2), cotizaciones (6) y devoluciones (4), más 128 documentos anulados. El modelo se entrenaba, entonces, con una demanda que no era venta.

### Cambio de SQL

`sql/extraccion_ventas_v6.sql` es idéntica a la V5 (columnas, orden, `GROUP BY`, filtros y `HAVING`) más `d.IDTIPODOCINV IN (1, 13)` y `d.ANULADO = '0'`. Las devoluciones en ventas (tipo 4) se **excluyen** y no se restan: son 30 documentos, 4.123.300 COP y 58 unidades en el periodo (0,06 % del monto V6). Nuevo extractor offline `sql/extraer_ventas.py` (`--db --sql --out --desde --hasta`; cliente Firebird embebido por `FIREBIRD_CLIENT`, usuario y clave por variables de entorno) que escribe el CSV en el formato exacto de la V5; el cargador existente lo lee sin cambios. Con él se generaron `data/raw/Query_Result_V6.csv` (misma ventana que la V5) y `data/holdout/ventas_2026_04_09.csv` (ventas posteriores, solo para el estudio de caso fuera de muestra). `Query_Result_V5.csv` pasó a `data/historico/` y `paths.DEFAULT_DATASET` apunta a la V6. Las extracciones se hicieron sobre una copia de la base; la original no se tocó.

### Dataset V5 vs V6 (2024-01-09 a 2026-03-27)

| | V5 | V6 | V6 / V5 |
|---|---:|---:|---:|
| Filas (producto-día) | 78.921 | 48.517 | 61,5 % |
| Productos | 5.654 | 3.576 | 63,2 % |
| Monto total (COP) | 13.323.312.035 | 6.741.707.924 | 50,6 % |
| Filas excluidas por `remove_bulk_adjustments` | 1.761 (1 fecha: 19/02/2026, 2.345.652 COP) | 0 | |
| Productos tras limpieza | 5.074 | 3.576 | |

El caso del 19/02/2026 era mayormente de pedidos y ajustes y desaparece con el filtro de tipo; `remove_bulk_adjustments` se conserva, pero ya no elimina nada en la V6. El holdout (2026-03-28 a 2026-09-22, últimos datos de la base): 13.807 filas, 2.465 productos, 3.029.726.979 COP.

Monto mensual: la V6 es 20-38 % de la V5 entre feb-2024 y abr-2025 y 50-73 % desde may-2025 (p. ej. feb-2024: 463,1 M vs 91,7 M; jul-2025: 1.061,6 M vs 765,7 M).

### Caída de facturación feb-2024 a abr-2025 (solo documentación)

Las facturas pasan de ~1.500 a ~450 por mes entre feb-2024 y abr-2025 y se recuperan en may-2025. Decisión del usuario: se mantiene el mismo periodo que la V5 y ese tramo se usa tal cual. Verificación contable (cuenta de ingresos 4135.06, crédito menos débito, asientos no anulados) frente a facturas 1+13, en millones de COP:

| Mes | Ingresos contables | Facturas |
|---|---:|---:|
| 2023-12 | 252,8 | 253,1 |
| 2024-01 | 278,0 | 278,4 |
| 2024-02 | 29,6 | 91,7 |
| 2024-03 | 56,6 | 56,6 |
| 2024-06 | 70,7 | 75,5 |
| 2024-10 | 104,5 | 112,1 |
| 2025-03 | 97,6 | 101,3 |
| 2025-04 | 160,3 | 162,8 |
| 2025-05 | 354,7 | 357,8 |
| 2025-07 | 764,3 | 765,7 |

La contabilidad cae igual que las facturas (salvo feb-2024, donde las facturas superan a lo contabilizado): no es un hueco de registro del inventario. La base no permite distinguir una caída real de ventas de ventas registradas fuera del sistema.

### Antes (V5) y después (V6) del pipeline completo

Los números "antes" salen de los reportes previos (`eda_resumen.txt`, `seleccion_k_kmeans.csv`, `tuning_rf_resumen.txt`, `evaluacion_modelo.txt`, `comparacion_modelos.txt`); los "después", de los reportes regenerados con la V6 (`scripts/eda.py`, `select_k.py`, `tune_rf.py`, `evaluate_model.py`, `compare_models.py`). La razón suma predicha/real no está en los reportes: se calculó con `reg_test_results` de `train_regressor()` (V5: 0,5585, coincide con la entrada del 2026-10-02).

**EDA**

| | V5 | V6 |
|---|---:|---:|
| Filas producto-mes tras calendario completo | 99.765 | 66.236 |
| Productos tras limpieza | 5.074 | 3.576 |
| Producto-mes con venta cero | 66,53 % | 65,53 % |
| Mediana de meses en cero por producto (Q1 / Q3) | 76,47 % (50,00 / 88,89) | 75,00 % (51,85 / 87,50) |
| Pareto A / B / C (productos) | 532 (10,5 %) / 960 (18,9 %) / 3.582 (70,6 %) | 385 (10,8 %) / 661 (18,5 %) / 2.530 (70,7 %) |
| Ventas de A / B / C | 79,99 / 15,01 / 5,01 % | 79,99 / 15,01 / 5,00 % |

La intermitencia y el Pareto se mantienen casi iguales. El perfil estacional por mes cambia de nivel pero conserva el máximo en julio (promedio de julio: 681,1 M en V5 y 436,9 M en V6).

**Selección de k (silueta)**

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V5 | 0,3557 | 0,3753 | 0,3297 | 0,3361 | 0,3402 | 0,3508 | 0,3178 | 0,3220 | 0,3237 |
| V6 | 0,3598 | 0,3940 | 0,3427 | 0,3572 | 0,3588 | 0,3687 | 0,3433 | 0,3488 | 0,3440 |

k = 3 sigue teniendo la mayor silueta (0,3940).

**Ajuste de hiperparámetros del clasificador**

| | V5 | V6 |
|---|---|---|
| Mejores parámetros | `max_depth` 14, `min_samples_leaf` 3, `n_estimators` 300 | idénticos |
| F1 macro en validación cruzada | 0,4403 ± 0,0195 | 0,4366 ± 0,0131 |
| F1 macro de prueba, por defecto / ajustado | 0,4544 / 0,4525 | 0,4507 / 0,4507 |
| Acc. balanceada de prueba, por defecto / ajustado | 0,5087 / 0,5016 | 0,5166 / 0,5166 |

`DEFAULT_CLF_PARAMS` no cambia (ya coincide con los mejores parámetros; el reporte V5 mostraba `min_samples_leaf` 5 como "por defecto" por ser anterior a la actualización del 2026-09-23).

**Clasificador (prueba)**

| | V5 | V6 |
|---|---:|---:|
| n prueba | 8.942 | 6.255 |
| Distribución Reducir / Mantener / Reforzar | 78,42 / 3,81 / 17,77 % | 80,18 / 3,13 / 16,69 % |
| F1 macro | 0,4525 | 0,4507 |
| Accuracy balanceada | 0,5016 | 0,5166 |
| Accuracy | 0,6751 | 0,6614 |
| F1 ponderado | 0,7083 | 0,7089 |
| F1 Reducir / Mantener / Reforzar | 0,79 / 0,08 / 0,49 | 0,78 / 0,10 / 0,47 |
| Línea base mayoritaria: F1 macro / acc. balanceada / accuracy / F1 ponderado | 0,2930 / 0,3333 / 0,7842 / 0,6893 | 0,2967 / 0,3333 / 0,8018 / 0,7135 |

Matriz de confusión V6 (filas = real; orden Reducir, Mantener, Reforzar): [3427, 568, 1020], [77, 45, 74], [250, 129, 665]. La clase "Mantener" sigue siendo la débil (precisión 0,06, recall 0,23).

**Regresor (prueba, top productos)**

| | V5 | V6 |
|---|---:|---:|
| n prueba | 5.062 | 4.482 |
| MAE modelo / ingenuo / media_3 | 385.016 / 644.901 / 493.789 | 244.042 / 431.137 / 315.215 |
| RMSE modelo / ingenuo / media_3 | 1.244.584 / 1.651.228 / 1.232.872 | 973.944 / 1.273.318 / 929.992 |
| R² modelo / ingenuo / media_3 | 0,6005 / 0,2969 / 0,6080 | 0,4744 / 0,1016 / 0,5208 |
| MASE modelo / ingenuo / media_3 | 0,8887 / 1,4885 / 1,1397 | 1,0748 / 1,8988 / 1,3883 |
| Razón suma predicha / suma real | 0,5585 | 0,6717 |

El MASE del modelo en V6 (1,0748) supera 1 (el error de prueba es mayor que el error ingenuo dentro de la muestra de entrenamiento), aunque sigue siendo el menor frente a ingenuo y media_3; media_3 supera al modelo en RMSE y R², como ya ocurría en R² con la V5.

**Comparación de modelos**

| Clasificador | RF V5 | RF V6 | XGBoost V5 | XGBoost V6 | Mayoritaria V5 | Mayoritaria V6 |
|---|---:|---:|---:|---:|---:|---:|
| F1 macro | 0,4525 | 0,4507 | 0,4451 | 0,4350 | 0,2930 | 0,2967 |
| Acc. balanceada | 0,5016 | 0,5166 | 0,5116 | 0,5190 | 0,3333 | 0,3333 |
| Accuracy | 0,6751 | 0,6614 | 0,6190 | 0,6096 | 0,7842 | 0,8018 |
| F1 ponderado | 0,7083 | 0,7089 | 0,6807 | 0,6790 | 0,6893 | 0,7135 |

| Regresión A (prueba completa) | RF V5 / V6 | XGBoost V5 / V6 | Ingenuo V5 / V6 | Media_3 V5 / V6 |
|---|---:|---:|---:|---:|
| MAE | 385.016 / 244.042 | 428.863 / 258.181 | 644.901 / 431.137 | 493.789 / 315.215 |
| RMSE | 1.244.584 / 973.944 | 1.440.961 / 1.041.222 | 1.651.228 / 1.273.318 | 1.232.872 / 929.992 |
| R² | 0,6005 / 0,4744 | 0,4645 / 0,3993 | 0,2969 / 0,1016 | 0,6080 / 0,5208 |
| MASE | 0,8887 / 1,0748 | 0,9899 / 1,1371 | 1,4885 / 1,8988 | 1,1397 / 1,3883 |

| Regresión B (muestra; V5 n=371, V6 n=339) | RF V5 / V6 | XGBoost V5 / V6 | Ingenuo V5 / V6 | Media_3 V5 / V6 | ARIMA V5 / V6 | Prophet V5 / V6 |
|---|---:|---:|---:|---:|---:|---:|
| MAE | 296.272 / 146.781 | 344.308 / 154.674 | 451.419 / 311.407 | 367.612 / 234.136 | 320.998 / 214.765 | 310.613 / 191.777 |
| RMSE | 1.050.444 / 648.571 | 1.167.985 / 661.352 | 1.018.441 / 917.584 | 917.032 / 701.747 | 944.640 / 821.210 | 927.276 / 708.384 |
| R² | 0,5085 / 0,0902 | 0,3924 / 0,0539 | 0,5380 / -0,8211 | 0,6254 / -0,0652 | 0,6025 / -0,4587 | 0,6170 / -0,0854 |
| MASE | 0,6838 / 0,6465 | 0,7947 / 0,6812 | 1,0419 / 1,3715 | 0,8485 / 1,0312 | 0,7409 / 0,9459 | 0,7169 / 0,8446 |

El orden por MAE y MASE se mantiene (RF mejor; luego XGBoost y los modelos de serie de tiempo; ingenuo al final). En la regresión B con la V6 solo RF y XGBoost conservan R² positivo.

### Qué cambia en la narrativa

- La intermitencia (65,5 % de producto-mes en cero) y el Pareto (cerca de 11 % de los productos reúnen 80 % de las ventas) no dependían de la mezcla de documentos.
- El clasificador conserva su F1 macro (0,45) y supera con claridad a la línea base mayoritaria (0,30); Mantener sigue siendo la clase débil.
- Los errores absolutos del regresor bajan (la escala de ventas es menor) y el sesgo a la baja de la suma disminuye (razón 0,67 frente a 0,56), pero el MASE queda sobre 1 y media_3 supera al modelo en RMSE y R².
- k = 3 y los hiperparámetros se mantienen; no hay decisión pendiente sobre ellos.

## [Reestructuración] - 2026-10-03 (organización del repositorio)

Reorganización sin cambios en la lógica del modelo: las métricas de `outputs/reports/evaluacion_modelo.txt` son idénticas antes y después. Las entradas siguientes de este historial (y `docs/BITACORA.md`) citan las rutas antiguas tal como eran en su fecha.

| Antes | Ahora |
|---|---|
| `scripts/train_kmeans_rf_prod.py` | `src/ventas_forecast/model.py` (la ejecución directa pasó a `scripts/entrenar.py`) |
| `scripts/app.py`, `scripts/dashboard.html` | `dashboard/app.py`, `dashboard/dashboard.html` |
| `scripts/Iniciar_Dashboard.bat` | `Iniciar_Dashboard.bat` (raíz) |
| `scripts/requirements_dashboard.txt` | `requirements.txt` (modelo y dashboard); XGBoost, Prophet y statsmodels en `requirements-comparacion.txt` para que el lanzador no instale dependencias pesadas |
| `scripts/uploads/` (versionado) | `outputs/uploads/` (ignorado) |
| `scripts/train_kmeans_rf.py`, `run_xgboost.py`, `run_prophet.py`, `monitor_prophet.py`, `generar_reportes.py`, `debug_capture.py` | `legacy/scripts/` |
| `models/legacy/*` | `legacy/models/` |
| `tests/test_xgboost_simple.py`, `tests/test_xgboost_import.py` | `legacy/tests/` |
| `data/raw/Query_Result.csv`, `_V2`, `_V3`, `_V4`, `ventas.csv`, `cleaner.py` | `data/historico/` |
| `scripts_db/query_v5.sql` (sin versionar) | `sql/extraccion_ventas_v5.sql` |
| `scripts_db/consultas_sql*.txt` (sin versionar) | `sql/consultas_versiones_anteriores.sql` |
| `scripts_db/explorar_bd*.py` (credenciales en el código) | `sql/exploracion/` (credenciales por variables de entorno) |
| `docs/INDICE_DOCUMENTACION.md`, `docs/VERSION_SUMMARY.md` | eliminados (métricas y rutas desactualizadas) |
| `scripts/README.md` | fusionado en `README.md` |

Otros cambios:

- `src/ventas_forecast/paths.py` agrega `DEFAULT_DATASET`, `DEFAULT_PREDICTION` y `UPLOADS`, y elimina constantes a carpetas inexistentes.
- La salida por defecto del modelo es `outputs/predictions/PREDICCION_MENSUAL.csv`; antes se escribía junto al CSV de entrada (dejaba predicciones dentro de `data/raw/`).
- Se dejan de versionar `.pyc` y las subidas del dashboard; `.gitignore` cubre archivos locales de herramientas y credenciales.
- Nueva prueba `tests/test_paths.py` (rutas por defecto).

## [Corrección] - 2026-10-02 (total del dashboard y gráfico de distribución)

### Total de ventas predichas comparable y presentado como estimación conservadora (`/api/summary`, `dashboard.html`)

**Problema** (captura del dashboard): predicción de 268 M COP y "-52 %" frente al mes anterior. Dos causas: (1) la línea histórica sumaba TODOS los productos (685,7 M en marzo), mientras el total predicho solo cubre los 848 productos con predicción de monto (558,9 M reales en marzo); (2) el regresor aprende `log1p(target)`, así que la **suma** de sus predicciones subestima el total real (ratio 0,559 en prueba).

**Cambios**: la línea histórica se calcula solo con los productos que tienen `ventas_predichas`, y `variacion` usa esos mismos productos. El KPI deja de mostrar el porcentaje de variación y presenta el total como "estimación conservadora".

**Alternativas evaluadas y descartadas** (misma partición de prueba):

| Enfoque | MAE por producto | MASE | Suma predicha / real |
|---|---|---|---|
| RF sobre log1p (actual) | 385.016 | 0,889 | 0,559 |
| RF sobre pesos sin log | 551.371 | 1,273 | 1,415 |
| Gradient boosting con pérdida Poisson | 519.741 | 1,200 | 1,156 |
| RF log1p + factor de calibración del total (3,08) | 385.016 | 0,889 | 1,722 |

El factor de calibración estimado en el entrenamiento sobrecorrige, y en validación por origen móvil varía entre 2,3 y 19,6, así que no es confiable. Los regresores sin sesgo en el total empeoran el error por producto (MASE > 1). Se mantiene el regresor actual; un pronóstico del total mensual del taller queda como trabajo futuro.

### Gráfico de distribución de decisiones desbordado

El doughnut usaba `responsive: false`, así que el canvas tomaba su tamaño por defecto (300 × 150 px) y se salía del contenedor de 150 × 150 px. Ahora usa `responsive: true` y `maintainAspectRatio: false` y se ajusta al contenedor.

## [Limpieza] - 2026-09-23

### Ajustes masivos de inventario CONFIRMADOS y excluidos (`remove_bulk_adjustments`, `src/ventas_forecast/data/cleaning.py`)

**Por qué**: la entrada anterior ([Comparación] 2026-09-23) dejaba pendiente de decisión la anomalía del 19/02/2026 (1.761 productos con exactamente 12 unidades a 111 COP cada uno). El dueño del negocio la **CONFIRMÓ como una carga/ajuste de inventario del sistema, no una venta real**: es estadísticamente imposible que 1.761 productos sin relación entre sí (tornillos, retenes, rodillos, etc.) se vendan el mismo día con idéntica cantidad e idéntico precio. Además, la limpieza debe vivir **dentro del pipeline** (no solo en el CSV curado a mano como `data/raw/cleaner.py`), porque a futuro el sistema leerá directamente la base de datos de la empresa.

**Regla implementada** (`remove_bulk_adjustments(df, date_col, id_col, qty_col, price_col, min_products=100)`, función de módulo en `src/ventas_forecast/data/cleaning.py`, con envoltorio `DatasetCleaner.remove_bulk_adjustments()` para reutilizarla en scripts de limpieza offline): elimina las filas cuya combinación EXACTA (fecha, cantidad, precio) es compartida por al menos `min_products` productos **distintos** ese mismo día (se cuentan productos distintos, no filas: varias filas del mismo producto con igual fecha/cantidad/precio no inflan el conteo). No se hardcodea ninguna fecha, cantidad ni precio.

**Por qué el umbral por defecto (100) es conservador** — verificado sobre `data/raw/Query_Result_V5.csv`, top 5 grupos (fecha, cantidad, precio) por número de productos distintos:

| # | Fecha | Cantidad | Precio (COP) | Productos distintos |
|---|---|---|---|---|
| 1 | 2026-02-19 | 12 | 111 | **1.761** (caso confirmado) |
| 2 | 2024-03-27 | 20 | 1.111 | 46 |
| 3 | 2024-03-27 | 10 | 1.111 | 29 |
| 4 | 2024-03-27 | 40 | 1.111 | 22 |
| 5 | 2024-04-12 | 1 | 100 | 19 |

El segundo grupo más grande (46 productos) queda muy por debajo del umbral de 100: con `min_products=100` la regla elimina **exactamente** el caso confirmado y no toca ningún otro grupo de la V5 actual.

**Integración en el pipeline** (`scripts/train_kmeans_rf_prod.py`): `SalesForecastModel.load_and_clean()` aplica la regla justo después de parsear los numéricos (`_parse_numeric`) y antes de agregar a producto-mes, porque el ajuste se detecta a nivel de **fila diaria** (columna `FECHA`, agregada como nuevo rol `"date"` en `DEFAULT_COL_MAP`) — a nivel mensual varios productos legítimos podrían coincidir por casualidad. Nuevo parámetro del constructor `bulk_adjustment_min_products=100` (`None` desactiva la regla); el resumen de filas eliminadas queda en `self.cleaning_summary` (rows_removed, distinct_products, affected_dates, cop_amount) y se imprime. `scripts/app.py` no necesita cambios: no pasa este parámetro, así que usa el valor por defecto.

**Pruebas nuevas** (TDD, RED→GREEN, `tests/test_bulk_adjustments.py`, 8 casos): elimina un grupo sintético que alcanza el umbral; conserva un grupo por debajo del umbral; conserva la misma cantidad/precio en fechas distintas (cada fecha se evalúa por separado); cuenta productos DISTINTOS y no filas (varias filas del mismo producto no inflan el conteo); `bulk_adjustment_min_products=None` desactiva la regla en el modelo; `load_and_clean()` la aplica de punta a punta sobre un CSV temporal con el formato real (separador `;`, números `12,00` / `1.332,00`).

**Resultados sobre `Query_Result_V5.csv`**:
- **1.761 filas eliminadas** (1.761 productos distintos, 1 fecha: 19/02/2026, **$2.345.652 COP** excluidos — 0,0176 % del total de ventas en pesos del dataset).
- HALLAZGO adicional: de esos 1.761 productos, **580 tenían esa fila de ajuste como su ÚNICO registro** en todo el histórico y por lo tanto **desaparecen por completo** del dataset limpio (5.654 → 5.074 productos distintos). Los 1.181 productos restantes conservan su historial real, solo sin esa fila.
- `outputs/reports/eda_resumen.txt` (sección 1, calidad de datos) ahora reporta explícitamente el ajuste excluido y los productos que desaparecen por completo.
- El pico irreal del histograma de distribución de ventas (`outputs/figures/eda_distribucion_ventas.png`) alrededor de ~1,3 k COP **desaparece** tras la exclusión (verificado visualmente).

**Recálculo de k, ajuste de hiperparámetros, evaluación, comparación y EDA sobre la V5 limpia** (antes = con el ajuste de inventario incluido, ver entradas anteriores; después = excluido):

**1) Selección de k** (`scripts/select_k.py`, `evaluate_k`) — k=3 **sigue siendo** el máximo de silueta, sin cambios de conclusión:

| k | Silueta (antes) | Silueta (después) |
|---|---|---|
| 2 | — | 0,3557 |
| **3** | **0,3782** | **0,3753** |
| 4 | — | 0,3297 |

**2) Ajuste de hiperparámetros** (`scripts/tune_rf.py`, `scoring=f1_macro`, 5 folds mensuales, 24 combinaciones, 149,3 s):

| | Antes | Después |
|---|---|---|
| Mejores hiperparámetros | `n_estimators=300, max_depth=14, min_samples_leaf=5` | `n_estimators=300, max_depth=14, min_samples_leaf=3` |
| F1 macro CV (media ± desv.) | 0,4415 ± 0,0178 | 0,4403 ± 0,0195 |

`min_samples_leaf` pasa de 5 a 3 (`n_estimators`/`max_depth` no cambian): un modelo con hojas ligeramente más pequeñas vuelve a ser el mejor una vez excluidos los 1.761 registros de ajuste. **`DEFAULT_CLF_PARAMS` actualizado** (`scripts/train_kmeans_rf_prod.py`) y **pinning test actualizado con TDD** (`tests/test_rf_tuning.py::test_default_clf_params_are_the_tuned_values`: RED con la aserción `min_samples_leaf == 3` contra el valor viejo `5` → GREEN al actualizar la constante).

**3) Evaluación del modelo** (`scripts/evaluate_model.py`):

Clasificador (prueba, n=8.942 antes de la limpieza tenía n=9.578; distribución de clases similar):

| Métrica | Antes | Después |
|---|---|---|
| F1 macro | 0,4606 | 0,4525 |
| Accuracy balanceada | 0,5148 | 0,5016 |
| Accuracy | 0,6831 | 0,6751 |
| F1 ponderado | 0,7194 | 0,7083 |
| F1 Reducir / Mantener / Reforzar | 0,80 / 0,10 / 0,48 | 0,79 / 0,08 / 0,49 |
| Línea base "siempre Reducir" — F1 macro / acc. bal. / Accuracy | 0,2948 / 0,3333 / 79,30 % | 0,2930 / 0,3333 / 78,42 % |

Ligero deterioro generalizado (menos de 1,2 puntos porcentuales en cada métrica): esperable, porque los 1.761 registros de ajuste creaban 1.761 observaciones "producto vende exactamente 12 unidades a 111 COP" artificialmente fáciles de clasificar de forma consistente; al quitarlas, el problema real (más ruidoso) queda mejor representado. **La conclusión no cambia**: el modelo sigue superando ampliamente a la línea base de clase mayoritaria en F1 macro y accuracy balanceada, y sigue por debajo en accuracy simple (mismo patrón que antes).

Regresor (prueba, top 1.900 productos, n=5.038 antes → n=5.062 después):

| Métrica | Modelo (antes) | Modelo (después) | Ingenuo (antes) | Ingenuo (después) | Media_3 (antes) | Media_3 (después) |
|---|---|---|---|---|---|---|
| MAE (COP) | 382.162 | 385.016 | 645.453 | 644.901 | 494.885 | 493.789 |
| RMSE (COP) | 1.224.574 | 1.244.584 | 1.654.083* | 1.651.228 | 1.235.545* | 1.232.872 |
| R² | 0,6150 | 0,6005 | 0,2976 | 0,2969 | 0,6081 | 0,6080 |
| MASE | 0,8743 | 0,8887 | 1,4766 | 1,4885 | 1,1322 | 1,1397 |

(*RMSE "antes" tomado de `compare_models.py`, no reportado en la entrada de evaluación original). **Cambio de conclusión parcial**: el modelo sigue superando a ambas líneas base en MAE y MASE (MASE < 1) y al ingenuo en todas las métricas, pero Media_3 pasa a tener RMSE y R² ligeramente mejores (RMSE 1.232.872 vs 1.244.584; R² 0,6080 vs 0,6005): Media_3 comete menos errores muy grandes, el modelo acierta mejor en el error típico.

**4) Comparación de modelos** (`scripts/compare_models.py`, 214,1 s):

Clasificador (n=8.942): Random Forest sigue ganando en F1 macro (0,4525 vs XGBoost 0,4451 vs mayoritaria 0,2930) y F1 ponderado; XGBoost sigue ganando por poco en accuracy balanceada (0,5116 vs 0,5016) — mismo patrón que antes, sin cambio de conclusión. Mejores hiperparámetros de XGBoost sin cambios: `learning_rate=0,05, max_depth=4, n_estimators=200`.

Regresión A (prueba completa, top productos, n=5.038 antes → n=5.062 después): Random Forest gana a XGBoost e ingenuo en las 4 métricas (MAE 385.016 vs XGBoost 428.863, R² 0,6005 vs 0,4645, MASE 0,8887 vs 0,9899) y a Media_3 en MAE y MASE (493.789 / 1,1397), pero **Media_3 gana en RMSE y R²** (1.232.872 / 0,6080 vs 1.244.584 / 0,6005) — cambio de conclusión respecto de la versión anterior, donde RF ganaba las 4.

Regresión B (muestra de 150 productos, semilla 42; n=348 filas antes → **n=371 filas después**, porque el universo de productos elegibles cambió tras la limpieza):

| Métrica | RF | XGBoost | Ingenuo | Media_3 | ARIMA(1,1,1) | Prophet |
|---|---|---|---|---|---|---|
| MAE (COP) | **296.272** | 344.308 | 451.419 | 367.612 | 320.998 | 310.613 |
| RMSE (COP) | 1.050.444 | 1.167.985 | 1.018.441 | **917.032** | 944.640 | 927.276 |
| R² | 0,5085 | 0,3924 | 0,5380 | **0,6254** | 0,6025 | 0,6170 |
| MASE | **0,6838** | 0,7947 | 1,0419 | 0,8485 | 0,7409 | 0,7169 |

**CAMBIO DE CONCLUSIÓN a declarar honestamente**: antes de esta limpieza, Random Forest ganaba en las 4 métricas de la Regresión B (MAE 265.580, RMSE 741.062, R² 0,2462, MASE 0,6076, superando también a ARIMA y Prophet). **Después de excluir los ajustes de inventario, Random Forest sigue ganando en MAE y MASE** (las métricas que este trabajo usa como principales para el regresor, ver entradas anteriores), **pero el baseline Media_3 pasa a tener mejor RMSE y R² que Random Forest** en esta muestra de 150 productos (RMSE 917.032 vs 1.050.444; R² 0,6254 vs 0,5085). El mismo patrón aparece, más atenuado, en la Regresión A (conjunto de prueba completo): RF gana en MAE y MASE y Media_3 en RMSE y R². Como el RMSE y el R² penalizan con fuerza los errores muy grandes, la lectura es que RF tiene menor error típico y Media_3 menos errores extremos. La diferencia en Regresión B es atribuible al tamaño de muestra (150 productos, 371 filas) y a que la semilla 42 ahora sortea sobre un universo de productos distinto (5.074 en vez de 5.654); se reporta sin ocultarla, para el capítulo de resultados del documento final.

**5) EDA** (`scripts/eda.py`, 15,9 s):

| | Antes | Después |
|---|---|---|
| Productos distintos (tras limpieza) | 5.654 | 5.074 |
| Producto-mes antes de completar calendario | 34.938 | 33.387 |
| Producto-mes tras completar calendario | 100.937 | 99.765 |
| Clase A / B / C (productos) | 532 (9,4 %) / 963 (17,0 %) / 4.159 (73,6 %) | 532 (10,5 %) / 960 (18,9 %) / 3.582 (70,6 %) |
| Clase A / B / C (% de ventas) | 79,97 % / 15,02 % / 5,00 % | 79,99 % / 15,01 % / 5,01 % |
| % producto-mes con venta cero | 65,39 % | 66,53 % |
| Mediana / Q1 / Q3 de venta-cero por producto | 71,43 % / 50,00 % / 86,36 % | 76,47 % / 50,00 % / 88,89 % |

El porcentaje de productos en clase C baja (menos productos "de relleno" sin ventas reales) y el share de meses en cero sube ligeramente (consistente con que los 580 productos que desaparecieron eran, por definición, los que menos historial real tenían). Los 3 segmentos de clúster (bajo volumen/precio, medio, alta rotación) se mantienen cualitativamente iguales, solo con índices de cluster reordenados por KMeans y valores medianos ligeramente distintos (p. ej. cluster de alta rotación: 339.500→353.114 COP, frecuencia mediana 11 sin cambios).

**Figuras verificadas visualmente** (Read): `eda_ventas_mensuales.png`, `eda_estacionalidad.png`, `eda_pareto_abc.png`, `eda_distribucion_ventas.png` (pico de ~1,3 k COP ya no aparece), `outputs/figures/comparacion_modelos.png`, `matriz_confusion.png`, `real_vs_predicho.png`, `seleccion_k_kmeans.png` — todas renderizan correctamente. Nota cosmética preexistente (no introducida por este cambio, no corregida aquí): en `comparacion_modelos.png` la etiqueta roja "Pronóstico ingenuo (entrenamiento)" se superpone parcialmente con las marcas del eje X del panel de MASE.

**Archivos modificados**: `src/ventas_forecast/data/cleaning.py` (nueva función `remove_bulk_adjustments` + método `DatasetCleaner.remove_bulk_adjustments`), `src/ventas_forecast/data/__init__.py` (export), `scripts/train_kmeans_rf_prod.py` (`DEFAULT_COL_MAP["date"]`, constructor `bulk_adjustment_min_products`, `self.cleaning_summary`, llamada en `load_and_clean()`, `DEFAULT_CLF_PARAMS` actualizado), `scripts/eda.py` (sección 1 reporta el ajuste excluido y los productos perdidos), `tests/test_bulk_adjustments.py` (nuevo), `tests/test_rf_tuning.py` (pinning test actualizado).

---

## [EDA] - 2026-09-23

### Análisis exploratorio de datos (`src/ventas_forecast/eda.py`, `scripts/eda.py`)

**Por qué**: las fases 2 (comprensión de datos) y 3 (preparación de datos) de CRISP-DM necesitan evidencia exploratoria explícita del trabajo de grado — calidad de datos, tamaño del dataset en cada etapa de limpieza, clasificación ABC, intermitencia de la demanda y estacionalidad — que hasta ahora solo existía implícita en el código de `SalesForecastModel`.

**Nuevas funciones puras** (`src/ventas_forecast/eda.py`, TDD con `tests/test_eda.py`, RED→GREEN, 12 casos):
- `data_quality_summary(raw_df, col_map)`: filas/columnas, nulos por columna, duplicados exactos, filas con id de producto nulo, número de productos y rango de fechas — sobre el CSV **crudo**, antes de cualquier agregación.
- `monthly_totals(df, date_col, value_col)` / `seasonal_profile(monthly)`: total por mes calendario y, a partir de esa serie, promedio por número de mes (1-12) junto con el número de AÑOS que aportaron una observación a ese mes (para exponer lo débil de la estimación con solo 2-3 años de historial).
- `abc_classification(df, id_col, value_col, a=0.80, b=0.95)`: tabla ABC por producto (participación acumulada descendente) y resumen (productos y % de ventas por clase). Regla de frontera DECIDIDA y probada explícitamente: la clase se define por el acumulado INCLUYENDO al propio producto (A si ≤80 %, B si ≤95 %, C en el resto); un producto que por sí solo hace que el acumulado SUPERE un corte (p. ej. uno que concentra el 82 % de las ventas) queda clasificado en la clase siguiente, no en la anterior.
- `intermittency_summary(df, id_col, value_col)`: fracción global de filas producto-mes con venta cero y distribución (mediana/Q1/Q3) de esa fracción por producto, sobre el calendario ya completo.

**Script** `scripts/eda.py`: usa el CSV crudo solo para la calidad de datos, y el pipeline real (`load_and_clean` → `build_features` → `cluster` → `prepare_split`, instrumentando `_complete_calendar` sin alterar su lógica) para todo lo demás. Genera `outputs/reports/eda_resumen.txt` y 4 figuras (paleta del proyecto, sin títulos dentro de la imagen): `eda_ventas_mensuales.png`, `eda_estacionalidad.png`, `eda_pareto_abc.png`, `eda_distribucion_ventas.png`.

**Resultados reales** (`Query_Result_V5.csv`, 18,0 s):

- **Calidad de datos (crudo)**: 78.921 filas, 13 columnas, 0 duplicados exactos. Único nulo: `CODIGO` en 8 filas (`DESCRIPCION="VARIOS"`, 435.000 COP, 0,0033 % del total). HECHO verificado en el código: `load_and_clean()` agrupa con `groupby([id, description, fecha])`, que usa `dropna=True` por defecto — esas 8 filas se excluyen silenciosamente del agregado y nunca llegan a clustering/entrenamiento/predicción (pérdida económica despreciable, pero silenciosa). 5.654 productos; rango 2024-01-09 a 2026-03-27 (27 meses distintos); primer mes (2024-01) parcial (arranca el día 9) y último mes (2026-03) parcial (termina el día 27).
- **Tamaño del dataset**: 78.921 filas crudas → 34.938 filas producto-mes (antes de completar calendario) → **100.937 filas** tras `_complete_calendar` (65.999 filas de mes sin venta agregadas), 5.654 productos, 27 meses calendario.
- **Clasificación ABC** (participación de `TOTAL_VENDIDO`, corte 80/95): Clase A 532 productos (9,4 %) → 79,97 % de las ventas; Clase B 963 productos (17,0 %) → 15,02 %; Clase C 4.159 productos (73,6 %) → 5,00 %. Concentración fuerte: menos del 10 % de los productos explica el 80 % de las ventas.
- **Intermitencia**: 65,39 % de las filas producto-mes tienen venta cero. Distribución por producto de esa fracción: mediana 71,43 %, Q1 50,00 %, Q3 86,36 % — la mayoría de los productos no vende en más de la mitad de sus meses activos, y para la mitad de los productos eso ocurre en más del 71 % de los meses.
- **Estacionalidad** (promedio mensual, ADVERTENCIA — solo 2-3 años de historial): mínimo en abril (344,1 M COP, n=2 años) y máximo en julio (681,1 M COP, n=2 años); enero/febrero/marzo tienen 3 años de historial (2024-2026), el resto solo 2 — estimación estacional débil, no concluyente para el documento final.
- **Perfil de clusters** (partición de entrenamiento, escala original, k=3): cluster 0 — 10.411 filas, mediana `TOTAL_VENDIDO` 120.000 COP, `FRECUENCIA` 2,0, `PRECIO_PROMEDIO` 29.850 COP; cluster 1 — 8.433 filas, 6.408 COP, 1,0, 1.200 COP (ventas y precio bajos, baja frecuencia); cluster 2 — 4.876 filas, 339.500 COP, 11,0, 13.336 COP (mayor frecuencia y ventas, precio intermedio). Interpretación cualitativa completa (nombres de segmento) se deja para el documento de tesis con estos números como evidencia.

## [Comparación] - 2026-09-23

### Comparación justa de modelos sobre V5 (`scripts/compare_models.py`, `src/ventas_forecast/benchmarks.py`)

**Condiciones de equidad**: misma partición temporal 80/20 por producto, mismas filas de prueba y mismas métricas para todos los modelos.
- Clasificación: Random Forest vs XGBoost vs clase mayoritaria. XGBoost se ajustó con GridSearchCV sobre los mismos pliegues mensuales y F1 macro, con pesos de clase balanceados (mejor: `learning_rate=0.05, max_depth=4, n_estimators=200`).
- Regresión A: las 5.038 filas de prueba de los productos principales. El regresor XGBoost no se ajustó (`n_estimators=400, max_depth=6, learning_rate=0.05`).
- Regresión B: muestra semillada de 150 productos (348 filas) para incluir Prophet y ARIMA(1,1,1), que se re-entrenan mes a mes con pronóstico a un paso (nunca ven el mes que pronostican). Respaldo ingenuo cuando el historial tiene menos de 6 meses: 9 filas en cada modelo.

**Corrección previa en Prophet**: la estacionalidad anual automática (se activa con ~2 años de datos) sobreajustaba series intermitentes y disparaba el pronóstico (hasta 4,2e10 frente a un máximo histórico de 1,3e6; MAE 122.510.008). Se desactivó (`yearly_seasonality=False`); prueba de regresión con la serie real del producto 1185.

**1) Clasificador (prueba, n = 9.578)**

| Métrica | Random Forest | XGBoost | Clase mayoritaria |
|---|---|---|---|
| F1 macro | **0,4606** | 0,4454 | 0,2948 |
| Accuracy balanceada | 0,5148 | **0,5185** | 0,3333 |
| Accuracy | 0,6831 | 0,6276 | 0,7930 |
| F1 ponderado | **0,7194** | 0,6905 | 0,7014 |
| F1 Reducir / Mantener / Reforzar | 0,80 / 0,10 / 0,48 | 0,76 / 0,10 / 0,47 | 0,88 / 0 / 0 |

Random Forest gana en F1 macro y F1 ponderado; XGBoost gana por poco en accuracy balanceada (+0,4 puntos). Ambos superan ampliamente a la clase mayoritaria en las métricas principales.

**2) Regresión A (prueba, n = 5.038)**

| Métrica | Random Forest | XGBoost | Ingenuo | Media 3 meses |
|---|---|---|---|---|
| MAE (COP) | **382.162** | 432.310 | 645.453 | 494.885 |
| RMSE (COP) | **1.224.574** | 1.441.172 | 1.654.083 | 1.235.545 |
| R² | **0,6150** | 0,4668 | 0,2976 | 0,6081 |
| MASE | **0,8743** | 0,9890 | 1,4766 | 1,1322 |

Random Forest gana en todas las métricas (XGBoost sin ajuste de hiperparámetros).

**3) Regresión B (muestra, n = 348)**

| Métrica | Random Forest | XGBoost | Ingenuo | Media 3 meses | ARIMA(1,1,1) | Prophet |
|---|---|---|---|---|---|---|
| MAE (COP) | **265.580** | 299.317 | 511.078 | 401.546 | 477.056 | 364.878 |
| RMSE (COP) | **741.062** | 763.001 | 1.010.047 | 790.182 | 1.782.513 | 1.018.184 |
| R² | **0,2462** | 0,2009 | −0,4004 | 0,1429 | −3,3615 | −0,4231 |
| MASE | **0,6076** | 0,6848 | 1,1692 | 0,9186 | 1,0914 | 0,8347 |

Random Forest gana en todas las métricas. ARIMA(1,1,1) presenta 3 pronósticos disparados en productos con historial corto (9 a 13 meses); sin esos 3 casos su MAE sería 353.010, y Random Forest seguiría siendo el mejor. No se recortaron sus predicciones.

Los scripts `scripts/run_xgboost.py` y `scripts/run_prophet.py` usan datasets antiguos y quedan sustituidos por `scripts/compare_models.py` para la comparación del trabajo de grado.


**Anomalía detectada — CONFIRMADA y excluida**: el 19/02/2026, 1.761 productos distintos registran exactamente 12 unidades a 111 COP (1.332 COP cada uno) — el pico del histograma de distribución. Representaba el 0,018 % de las ventas en pesos, pero ~5 % de los meses producto-mes con ventas. El dueño del negocio **CONFIRMÓ** que se trata de una carga/ajuste de inventario del sistema, no una venta real. Se excluyó del pipeline con una regla genérica (`remove_bulk_adjustments`) y se recalcularon k, hiperparámetros, evaluación, comparación y EDA — ver **[Limpieza] - 2026-09-23** (entrada más reciente, arriba) para la justificación del umbral, los resultados antes/después y si alguna conclusión cambió.

---

## [Mejora] - 2026-09-23 (métrica macro)

### Selección de hiperparámetros y métricas principales del clasificador: F1 macro / accuracy balanceada en vez de F1 ponderado / accuracy

**Por qué**: tras el completado de calendario (ver entrada anterior), la partición de prueba del clasificador queda muy desbalanceada — Reducir 7.595 filas (79,30 %), Reforzar 1.646 (17,19 %), Mantener 337 (3,52 %). Con ese desbalance, `tune_classifier(scoring="f1_weighted")` (el valor previo) selecciona hiperparámetros que maximizan el F1 ponderado por soporte, dominado por la clase mayoritaria "Reducir"; y reportar solo accuracy/F1 ponderado es engañoso: la línea base ingenua "siempre predecir Reducir" obtenía **Accuracy 79,30 %** y **F1 ponderado 70,14 %**, ambos por encima o cerca del clasificador entrenado (Acc 72,84 %, F1 74,26 %), pese a que esa línea base tiene macro F1 ≈ 0,29 (no distingue Mantener ni Reforzar en absoluto). DECIDIDO por el usuario: seleccionar hiperparámetros con **F1 macro** (promedia el F1 de cada clase con el mismo peso, sin importar su soporte) y reportar **F1 macro y accuracy balanceada** como métricas principales del clasificador; accuracy y F1 ponderado se conservan en los reportes como referencia, con una nota explicando por qué no bastan solas bajo este desbalance.

**Cambios**:
- `scripts/train_kmeans_rf_prod.py`:
  - `tune_classifier()`: `scoring` por defecto pasa de `"f1_weighted"` a `"f1_macro"` (parámetro explícito, se puede seguir pasando otro valor). Docstring actualizado explicando la razón del cambio.
  - `train_classifier()`: `self.metrics` conserva sus claves existentes (`accuracy`, `precision`, `recall`, `f1` — leídas sin cambios por `scripts/app.py`) y agrega `f1_macro` (`f1_score(average="macro", zero_division=0)`) y `balanced_accuracy` (`balanced_accuracy_score`), redondeadas a 4 decimales; se imprimen junto a las demás.
  - `DEFAULT_CLF_PARAMS`: recalculado con `tune_classifier(scoring="f1_macro")` sobre `Query_Result_V5.csv` (ver tabla de tuning abajo). `max_depth` pasa de `None` a `14` y `min_samples_leaf` de `1` a `5` (un modelo algo menos profundo y con hojas más grandes generaliza mejor en las clases minoritarias Mantener/Reforzar bajo este criterio).
- `scripts/evaluate_model.py`: la sección del clasificador muestra primero las métricas principales (F1 macro, accuracy balanceada), luego las ponderadas como referencia, la distribución de clases de la prueba, y una nota explicando por qué accuracy sola engaña bajo el desbalance. La línea base de clase mayoritaria ahora también reporta F1 macro y accuracy balanceada (además de accuracy y F1 ponderado, que se conservan).
- `scripts/tune_rf.py`: el resumen muestra el `scoring` usado por `GridSearchCV` (`f1_macro`), su media ± desviación en validación cruzada, y compara por defecto vs. ajustado tanto en F1 macro/accuracy balanceada (principal) como en las métricas ponderadas (referencia).

**Pruebas nuevas/actualizadas** (TDD, `tests/test_rf_tuning.py`):
- `test_default_scoring_is_f1_macro` / `test_explicit_scoring_overrides_default`: espían la llamada a `GridSearchCV` (mock sobre `train_kmeans_rf_prod.GridSearchCV`) para verificar que `tune_classifier()` usa `scoring="f1_macro"` por defecto y respeta un `scoring` explícito.
- `ClassifierMetricsTest`: `self.metrics` conserva las claves ponderadas existentes y agrega `f1_macro`/`balanced_accuracy` con valores iguales a los calculados independientemente por `sklearn` sobre las mismas predicciones de prueba.
- `test_default_clf_params_are_the_tuned_values` (pinning test) y `test_custom_clf_params_are_applied_to_classifier`: actualizados (RED→GREEN) para los nuevos valores de `DEFAULT_CLF_PARAMS` (`max_depth=14`, `min_samples_leaf=5`).

**Resultados reales** (`scripts/tune_rf.py`, `scripts/evaluate_model.py` sobre `Query_Result_V5.csv`, misma partición temporal 80/20 por producto; 24 combinaciones evaluadas, 5 folds mensuales, scoring=f1_macro, 154,8 s):

- Mejores hiperparámetros: `n_estimators=300, max_depth=14, min_samples_leaf=5` (antes: `max_depth=None, min_samples_leaf=1`).
- F1 macro en validación cruzada (media ± desviación): **0,4415 ± 0,0178**.

Clasificador (prueba, n=9.578 — Reducir 7.595 / Mantener 337 / Reforzar 1.646):

| Métrica | Anterior (max_depth=None, min_samples_leaf=1) | Nuevo (max_depth=14, min_samples_leaf=5) |
|---|---|---|
| Accuracy | 72,84 % | 68,31 % |
| Accuracy balanceada | 48,73 % | **51,48 %** |
| F1 (ponderado) | 74,26 % | 71,94 % |
| F1 macro | 45,27 % | **46,06 %** |

Por clase F1 (prueba):

| Clase | Anterior | Nuevo |
|---|---|---|
| Reducir | 0,83 | 0,80 |
| Mantener | 0,05 | 0,10 |
| Reforzar | 0,48 | 0,48 |

Línea base "siempre Reducir" (clase mayoritaria de entrenamiento): Accuracy 79,30 %, F1 ponderado 70,14 %, **F1 macro 29,48 %**, **accuracy balanceada 33,33 %**.

**El modelo ajustado supera la línea base en F1 macro (46,06 % vs. 29,48 %) y en accuracy balanceada (51,48 % vs. 33,33 %), pero sigue por debajo en accuracy simple (68,31 % vs. 79,30 %)** — esperado y consistente con la razón del cambio: la línea base gana en accuracy únicamente porque ignora por completo las clases minoritarias, que es justo lo que F1 macro/accuracy balanceada penalizan y lo que el modelo, aunque de forma modesta, sí logra distinguir (recall de Mantener y Reforzar sube frente a la línea base, que tiene recall 0 en ambas). La clase "Mantener" sigue siendo la más débil del modelo (F1 0,10, 337 casos), aunque mejora respecto al ajuste anterior (0,05).

El regresor no depende del clasificador y sus métricas no cambian: MAE 382.161,91, R² 0,6150, MASE 0,8743 (ver `outputs/reports/evaluacion_modelo.txt`).

## [Corrección] - 2026-09-23

### Fuga de calendario: `shift()`/`rolling()` operaban por fila, no por mes calendario

**Problema**: `load_and_clean()` agrega el CSV crudo a una fila por producto-mes, pero **solo para los meses con ventas**: un mes sin ninguna venta no tiene fila. `build_features()` calcula `target`, `lag_k`, `mean_3`, `mean_6` y `media_3_actual` con `groupby(id).shift(...)`/`rolling(...)`, que operan sobre la POSICIÓN de la fila dentro de cada producto, no sobre el mes calendario. En `Query_Result_V5.csv`, el **40,2 % de las filas (11.764 de 29.284)** tenían un salto de más de 1 mes hasta la fila siguiente del mismo producto. Consecuencias:
- `target` no era "ventas del mes calendario siguiente" sino "ventas del siguiente mes CON ventas" (podía saltarse meses en cero).
- `lag_k`, `mean_3`, `mean_6`, `media_3_actual` eran "los k meses con ventas anteriores", no los k meses calendario anteriores.
- Los meses con venta 0 no existían como observación: un producto que deja de venderse era invisible para la clase "Reducir" (nunca se generaba un `target=0` correspondiente a un mes calendario en cero).

**Corrección** (`scripts/train_kmeans_rf_prod.py`), reglas aprobadas por el usuario (2026-09-23):
- Nuevo método `_complete_calendar(df)`, llamado desde `load_and_clean()` justo después de la agregación (y de fijar `self._ultimo_mes`/`self._mes_pred`): completa el calendario de cada producto con una fila por mes, desde su PRIMER mes con ventas hasta el último mes GLOBAL del dataset (`self._ultimo_mes`), rellenando `sales`, `total_sold` y `frequency` con 0 en los meses sin ventas. `avg_price` se rellena con forward-fill (el último precio conocido del producto), porque no hay ninguna transacción de la que derivar un precio propio ese mes; `description` también se forward-fillea; `year`/`month` se recalculan de `fecha`. Imprime cuántas filas de venta 0 se agregaron.
- Con el calendario completo, `target`/`lag_k`/`mean_3`/`mean_6`/`media_3_actual` (sin cambios de código en `build_features()`) pasan automáticamente a ser "por mes calendario", incluyendo correctamente los ceros.
- Un mes calendario siguiente en 0 ahora produce naturalmente `growth_ratio = 0 < reduce_threshold` ⇒ clase "Reducir" (sin necesidad de una regla especial).
- `_build_train_frame()`: se elimina el filtro `target > 0` (queda `target.notna() & (total_sold > 0)`); un mes siguiente en cero es una observación válida y necesaria para entrenar "Reducir". Solo se excluyen las filas cuyo MES ACTUAL no tuvo ventas (`total_sold == 0`), porque `growth_ratio` quedaría indefinido (división por cero) — usado por `cluster()`, `evaluate_k()` y `prepare_split()`.
- `train_regressor()`: `tiene_target` pasa de `target > 0` a `target.notna()` (consistente con lo anterior); el regresor ahora entrena y evalúa también sobre `target=0` (`log1p(0)=0` no da problema numérico). La definición de MASE y de los baselines ingenuo/media_3 no cambia.
- `prepare_split()`: `min_months` ahora cuenta meses **con ventas** (`total_sold > 0`) por producto, no filas de calendario — con el calendario completo, contar filas de calendario sobreestimaría el historial real de un producto con huecos. Las filas del último mes con `total_sold == 0` siguen excluidas de la predicción, igual que antes.
- `DEFAULT_CLF_PARAMS`: recalculado con `tune_classifier()` sobre los datos ya con calendario completo (ver más abajo); `max_depth` pasa de `14` a `None`.
- `scripts/app.py` no necesitó cambios: sigue llamando a los mismos métodos en el mismo orden.

**Pruebas nuevas**: `tests/test_monthly_calendar.py` (9 casos): una fila con mes faltante recibe `total_sold=0` y precio forward-filled; no quedan huecos de calendario (filas consecutivas por producto difieren en exactamente 1 mes); el calendario arranca en el primer mes con ventas de cada producto y termina en el último mes global; el `target` del mes anterior a un mes en cero es 0 y su `target_class` es 0 (Reducir) en `_df_train`; `lag_1` es el `total_sold` del mes CALENDARIO anterior (0 si ese mes no tuvo ventas); las filas con `total_sold` actual en 0 no entran a `_df_train`; `min_months` cuenta meses con ventas y no filas de calendario. `tests/_helpers.py`: `synthetic_history()` gana un parámetro opcional `drop_months` para simular productos con huecos sin romper las llamadas existentes (por defecto no omite ningún mes). `tests/test_rf_tuning.py`: se actualiza `test_default_clf_params_are_the_tuned_values` (RED→GREEN, TDD) para esperar `max_depth=None` en vez de `14`, junto con el cambio de `DEFAULT_CLF_PARAMS`.

**Nota metodológica**: las cifras de la corrección anterior (clasificador Acc 64,12 % / Prec 63,93 % / Rec 64,12 % / F1 64,02 %; regresor MAE 460.968 / RMSE 1.288.345 / R² 0,6114 / MASE 1,0954; baselines ingenuo MAE 616.270 / R² 0,3793 / MASE 1,4645, media_3 MAE 531.697 / R² 0,5851 / MASE 1,2635) quedan **superadas**: se calcularon sobre una serie por fila (sin calendario), no por mes calendario, y por lo tanto no son comparables con las siguientes.

**Selección de k tras el completado de calendario** (`scripts/select_k.py`, sobre la partición de entrenamiento, ahora 23.720 filas en vez de 16.368-ish previas):

| k | Inercia | Silueta |
|---|---|---|
| 2 | 21.873,39 | 0,3597 |
| **3** | **14.473,17** | **0,3782** |
| 4 | 11.730,33 | 0,3275 |
| 5 | 9.610,21 | 0,3348 |
| 6 | 8.212,69 | 0,3360 |
| 7 | 7.234,13 | 0,3457 |
| 8 | 6.483,72 | 0,3165 |
| 9 | 5.904,14 | 0,3212 |
| 10 | 5.437,75 | 0,3223 |

k=3 **sigue siendo el máximo de silueta** (0,3782, antes 0,3695); no se modifica `n_clusters` (sigue en 3).

**Ajuste de hiperparámetros tras el completado de calendario** (`scripts/tune_rf.py`, misma grilla y `TimeSeriesSplit` mensual):

- Mejores hiperparámetros: `max_depth=None, min_samples_leaf=1, n_estimators=300` (antes: `max_depth=14`). **`DEFAULT_CLF_PARAMS` se actualiza** a `max_depth=None`.
- F1 ponderado en validación cruzada (media ± desviación): **0,6400 ± 0,0159** (antes 0,6041 ± 0,0138).
- Métricas de prueba (holdout, promedio ponderado):

| Métrica | Por defecto (max_depth=14) | Ajustado (max_depth=None) |
|---|---|---|
| Accuracy | 70,27 % | 72,84 % |
| Precision | 78,59 % | 77,63 % |
| Recall | 70,27 % | 72,84 % |
| F1 | 72,96 % | 74,26 % |

**Evaluación completa tras el completado de calendario** (`scripts/evaluate_model.py`, top 1.900 productos, 16.242 filas de entrenamiento / 5.038 de prueba para el regresor; 9.578 filas de prueba para el clasificador — antes 5.077 y ~4.756 respectivamente, porque el calendario completo agrega muchas más filas con `target`/`total_sold` válidos):

Clasificador (prueba, promedio ponderado):

| Métrica | Anterior (sin calendario) | Nuevo (con calendario) |
|---|---|---|
| Accuracy | 64,12 % | 72,84 % |
| Precision | 63,93 % | 77,63 % |
| Recall | 64,12 % | 72,84 % |
| F1 | 64,02 % | 74,26 % |

Línea base de clase mayoritaria (ahora "Reducir", antes "Reforzar" — el completado de calendario agrega muchísimas filas cuyo mes siguiente es cero, así que "Reducir" pasa a ser la clase ampliamente dominante en entrenamiento):

| Métrica | Modelo | Línea base (clase mayoritaria = "Reducir") |
|---|---|---|
| Accuracy | 72,84 % | **79,30 %** |
| F1 | 74,26 % | 70,14 % |

Reporte por clase (prueba):

```
              precision    recall  f1-score   support
     Reducir       0.89      0.78      0.83      7595
    Mantener       0.08      0.04      0.05       337
    Reforzar       0.38      0.65      0.48      1646
```

Matriz de confusión (filas = real, columnas = predicha; orden Reducir/Mantener/Reforzar):

```
Reducir  : [5895,  107, 1593]
Mantener : [ 152,   12,  173]
Reforzar : [ 550,   26, 1070]
```

Regresor (prueba, top productos, `n_test=5.038`):

| Métrica | Modelo | Ingenuo | Media_3 |
|---|---|---|---|
| MAE | 382.161,91 | 645.453,24 | 494.885,24 |
| MSE | 1,4996e+12 | 2,7360e+12 | 1,5266e+12 |
| RMSE | 1.224.573,75 | 1.654.083,12 | 1.235.545,14 |
| R² | 0,6150 | 0,2976 | 0,6081 |
| MASE | **0,8743** | 1,4766 | 1,1322 |

**¿El modelo sigue superando cada línea base?** Dicho sin adornos:
- **Regresor: sí, en las cinco métricas** frente a ambos baselines (menor MAE/MSE/RMSE/MASE, mayor R²). Además, a diferencia de la corrección anterior, el **MASE del modelo ahora es 0,8743 (< 1)**: el error absoluto medio del modelo en prueba es menor que el error absoluto medio del pronóstico ingenuo observado en entrenamiento (la escala de MASE), no solo menor que el ingenuo evaluado en las mismas filas de prueba.
- **Clasificador: en F1 sí (74,26 % vs 70,14 %), pero en Accuracy NO** — la línea base de "siempre predecir Reducir" alcanza 79,30 % de accuracy, **5 puntos por encima del modelo (72,84 %)**. Esto ocurre porque el calendario completo hace que "Reducir" (target=0 o caída) sea abrumadoramente la clase mayoritaria (7.595 de 9.578 filas de prueba, 79 %); un clasificador trivial que siempre predice "Reducir" ya acierta esa proporción. El modelo sacrifica accuracy global a cambio de detectar mejor las clases minoritarias "Reforzar" (recall 0,65 vs 0 del baseline) y, débilmente, "Mantener" (recall 0,04): eso es lo que refleja el F1 ponderado más alto, no la accuracy.

**Concernientes a documentar honestamente**:
- La clase "Mantener" es ahora aún más débil que antes (F1 0,05, recall 0,04, solo 337 casos de soporte en prueba): el modelo casi nunca la predice correctamente.
- La comparación de accuracy contra la línea base de clase mayoritaria empeoró respecto a la corrección anterior (antes el modelo ganaba +22,2 puntos de accuracy; ahora pierde -6,46 puntos), precisamente porque el desbalance de clases se agravó al completar el calendario (más filas "Reducir" reales). El F1 ponderado sigue favoreciendo al modelo, pero no se debe citar solo la accuracy sin esta salvedad.
- Estos números son metodológicamente más correctos que los anteriores (calendario completo = comparación real mes a mes), pero exponen que el problema es más difícil de lo que parecía: antes la ausencia de meses en cero ocultaba parte de la dificultad real de distinguir "Reducir" de "Mantener".

---

## [Corrección] - 2026-09-22 (regresor)

### Fuga de datos en el regresor + métricas de regresión (MAE, RMSE, R², MASE) y baselines

**Problema**: `train_regressor()` seleccionaba `self._top_prods` (los `top_n_products` con más ventas) sumando `TOTAL_VENDIDO` sobre **todas** las filas de `_df_train`, incluyendo la partición de prueba, y luego entrenaba el `RandomForestRegressor` con **todas** esas filas (entrenamiento + prueba). El regresor nunca se evaluaba: no había MAE, MSE, RMSE, R² ni MASE, ni una línea base ingenua con la que compararlo (pendiente técnico señalado en la auditoría de 2026-09-22, ver `CLAUDE.md`).

**Cambio** (`scripts/train_kmeans_rf_prod.py`):
- `_top_prods` ahora se calcula únicamente con `self._df_train.loc[self._train_idx]` (partición de entrenamiento).
- El `RandomForestRegressor` se entrena únicamente con filas de entrenamiento de esos top productos (`target > 0`), sobre `self.scaler.transform(X)` y `log1p(target)`, igual que antes.
- Se evalúa en la partición de **prueba** de los mismos productos: predicciones revertidas a escala original con `expm1`. Se guarda `self.reg_metrics` (`mae`, `mse`, `rmse`, `r2`, `mase`, `n_test`) y `self.reg_test_results` (real vs. predicho, para graficar).
- **Definición de MASE**: `mase = mae_modelo / escala`, donde `escala = media(|target − TOTAL_VENDIDO|)` del pronóstico ingenuo (el mes actual predice el mes siguiente) calculada **solo sobre la partición de entrenamiento** de los top productos (no sobre prueba, para no filtrar información de prueba en la escala de referencia).
- Se agregan dos baselines evaluados sobre las mismas filas de prueba, guardados en `self.baseline_metrics`:
  - `ingenuo`: predice el mes siguiente igual al `TOTAL_VENDIDO` del mes actual.
  - `media_3`: predice la media móvil de 3 meses **incluyendo el mes actual** (columna nueva `media_3_actual`, sin `shift`; distinta de `mean_3`, que sí lleva `shift(1)` y se usa como feature del modelo). Si un producto no tiene historial suficiente (`NaN`), se usa el valor ingenuo como respaldo para esa fila.
- Nuevo script `scripts/evaluate_model.py`: corre el pipeline hasta `train_regressor()` sobre `Query_Result_V5.csv`, exporta `outputs/reports/evaluacion_modelo.txt` (métricas del clasificador + línea base de clase mayoritaria + matriz de confusión + tabla de regresión), `outputs/figures/matriz_confusion.png` y `outputs/figures/real_vs_predicho.png` (dispersión real vs. predicho, escala log-log).
- Tests nuevos en `tests/test_regression_metrics.py` (8 casos): verifican que el ajuste del regresor no usa filas de prueba (espiando `RandomForestRegressor.fit`), que `reg_metrics` tiene las claves esperadas con valores finitos y `rmse == sqrt(mse)`, que el MASE coincide con `mae / escala` calculada de forma independiente sobre entrenamiento, que el baseline `ingenuo` es coherente con un cálculo independiente sobre prueba, y que `_top_prods` ignora las filas de prueba (caso construido: se infla artificialmente `TOTAL_VENDIDO` solo en las filas de prueba de un producto para comprobar que NO pasa a ser top si la selección es correcta).

**Resultado sobre `Query_Result_V5.csv`** (`scripts/evaluate_model.py`, top 1900 productos, 16.368 filas de entrenamiento / 5.077 de prueba para el regresor):

Clasificador (prueba, promedio ponderado) — sin cambios respecto al ajuste de hiperparámetros ya adoptado:

| Métrica | Modelo | Línea base (clase mayoritaria = "Reforzar") |
|---|---|---|
| Accuracy | 64,12 % | 41,95 % |
| Precision | 63,93 % | — |
| Recall | 64,12 % | — |
| F1 | 64,02 % | 24,79 % |

Matriz de confusión (filas = real, columnas = predicha; orden Reducir/Mantener/Reforzar):

```
Reducir  : [2987,  176,  857]
Mantener : [ 233,   98,  345]
Reforzar : [ 903,  388, 2102]
```

Regresor (prueba, top productos, `n_test=5.077`):

| Métrica | Modelo | Ingenuo | Media_3 |
|---|---|---|---|
| MAE | 460.968,18 | 616.270,36 | 531.697,06 |
| MSE | 1.659.832.486.940,06 | 2.651.254.383.132,88 | 1.772.192.485.460,91 |
| RMSE | 1.288.344,86 | 1.628.267,29 | 1.331.237,20 |
| R² | 0,6114 | 0,3793 | 0,5851 |
| MASE | 1,0954 | 1,4645 | 1,2635 |

**El modelo supera a ambos baselines (ingenuo y media_3) en las cinco métricas** (menor MAE/MSE/RMSE/MASE, mayor R²) — dicho sin adornos, sí mejora frente a las líneas base ingenuas sobre la partición de prueba. El clasificador también supera ampliamente la línea base de clase mayoritaria (+22,2 puntos de accuracy, +39,2 puntos de F1 ponderado).

**Limitación a documentar honestamente**: el MASE del modelo es **1,0954, mayor a 1**. Esto significa que, aunque el modelo comete menos error absoluto que los baselines *sobre la partición de prueba*, su error absoluto medio en prueba es ligeramente **mayor** que el error absoluto medio del pronóstico ingenuo observado *en entrenamiento* (la escala de MASE). Es decir, el desempeño del pronóstico ingenuo se degrada más entre entrenamiento y prueba que el del modelo, pero ninguno de los dos "gana" en términos absolutos frente al comportamiento histórico de entrenamiento. No se debe presentar el MASE < 1 como si el modelo superara al ingenuo en términos absolutos: la comparación válida contra baselines es la de la tabla anterior (mismas filas de prueba para los tres), donde el modelo sí gana en las cinco métricas.

---

## [Mejora] - 2026-09-22 (adopción)

### Hiperparámetros ajustados como valores por defecto

`DEFAULT_CLF_PARAMS` pasa de `n_estimators=300, max_depth=10, min_samples_leaf=3` a `n_estimators=300, max_depth=14, min_samples_leaf=1` (mejor combinación de `tune_classifier()`).

Métricas de prueba sobre V5 con los nuevos valores por defecto: Accuracy 64,12 %, Precision 63,93 %, Recall 64,12 %, F1 64,02 %.

Nota metodológica: en la validación cruzada, las cinco mejores combinaciones quedan dentro de una desviación estándar (F1 0,6026–0,6041, σ ≈ 0,014); la mejora frente a los valores anteriores es moderada.

---

## [Mejora] - 2026-09-22 (ajuste de hiperparámetros)

### Ajuste de hiperparámetros del clasificador con validación cruzada (`GridSearchCV` + `TimeSeriesSplit` mensual)

**Problema**: los hiperparámetros del `RandomForestClassifier` (`n_estimators=300, max_depth=10, min_samples_leaf=3`) estaban fijados manualmente, sin ningún método de búsqueda ni validación cruzada (actividad A10 del anteproyecto pendiente).

**Cambio**:
- `SalesForecastModel` acepta `clf_params` en el constructor (`self.clf_params = {**DEFAULT_CLF_PARAMS, **clf_params}`); `train_classifier()` usa `self.clf_params` en vez de valores hardcodeados, así que el comportamiento por defecto no cambia si no se especifica `clf_params`.
- Nuevo método `_monthly_time_series_folds(dates, n_splits)`: aplica `TimeSeriesSplit` sobre los MESES únicos (no sobre las filas) para que ningún mes quede partido entre entrenamiento y validación, y todo mes de validación sea estrictamente posterior a los meses de entrenamiento del fold.
- Nuevo método `tune_classifier(param_grid=None, n_splits=5, scoring="f1_weighted")`: ejecuta `GridSearchCV` con un `Pipeline` (`RobustScaler` + `RandomForestClassifier`) sobre esos folds, usando **solo la partición de entrenamiento** (`self._train_idx`); el escalador se reajusta en cada fold para no filtrar estadísticos entre folds. Guarda `self.tuning_results` (tabla completa de `cv_results_`) y `self.best_clf_params`, sin modificar `self.clf_params` automáticamente.
- Nuevo script `scripts/tune_rf.py`: corre el pipeline completo, entrena con los hiperparámetros por defecto, ajusta con `GridSearchCV` y reentrena con los mejores hiperparámetros, comparando métricas de prueba (holdout) antes/después.

**Resultado sobre `Query_Result_V5.csv`** (grilla `n_estimators: [200, 300]`, `max_depth: [6, 10, 14, None]`, `min_samples_leaf: [1, 3, 5]`, 5 folds mensuales, `scoring=f1_weighted`, `scripts/tune_rf.py`):

- Mejores hiperparámetros: `max_depth=14, min_samples_leaf=1, n_estimators=300`.
- F1 ponderado en validación cruzada (media ± desviación): **0,6041 ± 0,0138**.
- Métricas de prueba (holdout, promedio ponderado):

| Métrica | Por defecto | Ajustado |
|---|---|---|
| Accuracy | 62,26 % | 64,12 % |
| Precision | 64,35 % | 63,93 % |
| Recall | 62,26 % | 64,12 % |
| F1 | 62,91 % | 64,02 % |

El ajuste mejora Accuracy, Recall y F1 (~+1,1 a +1,9 puntos porcentuales) y mantiene Precision prácticamente igual (-0,42 puntos). Tabla completa de la búsqueda: `outputs/reports/tuning_rf_clasificador.csv`; resumen: `outputs/reports/tuning_rf_resumen.txt`.

**Limitación**: la variable `cluster` (una de las features del clasificador) proviene del KMeans ajustado una sola vez sobre TODA la partición de entrenamiento (`cluster()`), no se refita dentro de cada fold de la validación cruzada; por lo tanto esa parte de la información (aunque solo del conjunto de entrenamiento, sin fuga hacia prueba) sí se comparte entre folds.

---

## [Mejora] - 2026-09-22

### Transformación logarítmica en el clustering y justificación de k=3

**Problema**: con las variables en escala original, la silueta máxima era k=2 (0,95), pero ese agrupamiento solo aislaba 72 registros de ventas extremas frente a 21.123; no representaba segmentos de productos.

**Cambio**: las variables de clustering (ventas, frecuencia, precio) se transforman con `log1p` antes del `RobustScaler` (`_cluster_features`), tanto en `cluster()` como en `evaluate_k()`. Se mantiene k=3.

**Selección de k sobre `Query_Result_V5.csv`** (partición de entrenamiento, `scripts/select_k.py`):

| k | Inercia | Silueta |
|---|---|---|
| 2 | 20.162,2 | 0,3493 |
| **3** | **13.248,3** | **0,3695** |
| 4 | 10.718,8 | 0,3225 |
| 5 | 8.786,5 | 0,3284 |
| 6 | 7.488,4 | 0,3285 |
| 7 | 6.563,4 | 0,3418 |
| 8 | 5.903,6 | 0,3142 |
| 9 | 5.356,2 | 0,3201 |
| 10 | 4.897,5 | 0,3196 |

k=3 obtiene la silueta máxima, coincide con el codo de la inercia y produce segmentos equilibrados (4.588 / 9.531 / 7.076 registros).

**Métricas del clasificador (promedio ponderado)**: Accuracy 62,16 → 62,26 %, Precision 64,16 → 64,35 %, Recall 62,16 → 62,26 %, F1 62,75 → 62,91 %.

---

## [Corrección] - 2026-09-22 (KMeans)

### Fuga de datos en el clustering + selección de k (`train_kmeans_rf_prod.py`)

**Problema**: `cluster()` ajustaba `RobustScaler` y `KMeans` con TODAS las filas de `self.df` (incluyendo el período de prueba y el mes a predecir), es decir, la misma fuga de datos que ya se había corregido en el escalador del clasificador. Además, `k=3` estaba fijado sin evidencia (sin método del codo ni coeficiente de silueta).

**Corrección**:
- Se extrajo el filtro de filas de entrenamiento y el split temporal por producto a un helper compartido (`_build_train_frame()`), usado tanto por `cluster()` como por `prepare_split()`, para que ambos entrenen con exactamente la misma partición.
- `cluster()` ahora ajusta el escalador (`self.cluster_scaler`) y KMeans (`self.kmeans`) solo con la partición de entrenamiento, y luego asigna cluster a todas las filas de `self.df` (prueba y mes a predecir) con `predict()`.
- Se agregó `evaluate_k(k_range, sample_size)`, que calcula inercia y silueta para un rango de k sobre la partición de entrenamiento (sin fuga), sin modificar `self.n_clusters` ni el estado del pipeline.
- Nuevo script `scripts/select_k.py` que ejecuta `evaluate_k` sobre `Query_Result_V5.csv` y exporta tabla + gráfico (codo y silueta).

**Resultado de `select_k.py` sobre `Query_Result_V5.csv`** (k evaluado de 2 a 10, sobre la partición de entrenamiento):

| k | Inercia | Silueta |
|---|---|---|
| 2 | 690563,58 | 0,9485 |
| 3 | 501512,56 | 0,8315 |
| 4 | 419882,47 | 0,7635 |
| 5 | 350324,75 | 0,7303 |
| 6 | 287355,06 | 0,6632 |
| 7 | 241275,30 | 0,6614 |
| 8 | 199370,31 | 0,6571 |
| 9 | 171041,06 | 0,6353 |
| 10 | 155281,90 | 0,6095 |

k con mayor silueta: **2** (por decidir: el modelo en producción se mantiene con `n_clusters=3`, valor que se conserva sin cambios hasta que se decida explícitamente si se ajusta).

**Impacto en el clasificador** (V5, promedio ponderado, 3 clases; la variable `cluster` es una de las features del clasificador, por eso cambia levemente al corregir la fuga del clustering):

| Métrica | Antes (solo fix de escalador) | Después (fix de KMeans) |
|---|---|---|
| Accuracy | 62,29 % | 62,16 % |
| Precision | 64,22 % | 64,16 % |
| Recall | 62,29 % | 62,16 % |
| F1 | 62,89 % | 62,75 % |

**Prueba**: `tests/test_kmeans_selection.py` (`python -m unittest tests.test_kmeans_selection`).

---

## [Corrección] - 2026-09-22

### Fuga de datos en el escalado (`train_kmeans_rf_prod.py`)

**Problema**: `RobustScaler` se ajustaba con todas las filas antes de la división temporal, por lo que las métricas de prueba incorporaban estadísticos (mediana e IQR) del conjunto de prueba.

**Corrección**: la división temporal por producto se calcula primero; el escalador se ajusta solo con las filas de entrenamiento y luego transforma todo el conjunto.

**Impacto medido sobre `Query_Result_V5.csv`** (promedio ponderado, 3 clases):

| Métrica | Antes | Después |
|---|---|---|
| Accuracy | 62,32 % | 62,29 % |
| Precision | 64,24 % | 64,22 % |
| Recall | 62,32 % | 62,29 % |
| F1 | 62,92 % | 62,89 % |

El impacto es mínimo porque Random Forest es invariante a transformaciones monótonas de escala; la corrección garantiza la validez metodológica de la evaluación.

**Prueba**: `tests/test_scaler_no_leakage.py` (`python -m unittest tests.test_scaler_no_leakage`).

---

## [Refactorización] - 2026-05-26

### 🎯 Refactorización de `train_kmeans_rf_prod.py` → Clase Reutilizable

**Cambio**: El script lineal se convirtió en la clase `SalesForecastModel`, parametrizable y aplicable a cualquier dataset de ventas.

**Antes**:
- Script monolítico de ~400 líneas
- Acoplado a `Query_Result_V3.csv`
- Nombres de columna hardcodeados
- Hiperparámetros fijos en el código

**Después**:
- Clase `SalesForecastModel` con 8 métodos encadenables
- Mapeo flexible de columnas mediante `col_map`
- Todos los hiperparámetros expuestos en el constructor
- Compatible con cualquier CSV de ventas
- Documentación completa en `README.md`

**API**:
```python
model = SalesForecastModel(
    filepath="datos.csv",
    col_map={"id": "product_id", "sales": "revenue", ...},
    numeric_fmt="plain"
)
model.run()  # Ejecuta todo
# O paso a paso
model.load_and_clean()
model.build_features()
# ... etc
```

**Beneficios**:
✅ Reutilizable en múltiples datasets  
✅ Sin modificar el código fuente  
✅ Hiperparámetros configurables  
✅ Pasos independientes y composables  
✅ Mejor testabilidad y mantenimiento  

**Métricas sin cambios**:
- Accuracy: 67.32%
- Precision: 60.25%
- Recall: 86.28%
- F1-Score: 70.95%

**Archivos modificados**:
- `scripts/train_kmeans_rf_prod.py` — refactorización completa
- `README.md` — documentación exhaustiva
- `docs/BITACORA.md` — detalles de la refactorización

---

## [Producción Final] - 2026-05-21

### 🎯 Modelo final — `train_kmeans_rf_prod`

**Script**: `scripts/train_kmeans_rf_prod.py`  
**Dataset**: `data/raw/Query_Result_V3.csv`  
**Salida**: `outputs/predictions/PREDICCION_MENSUAL.csv`

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.6732 |
| Precision | 0.6025 |
| Recall | 0.8628 |
| F1-Score | 0.7095 |

#### Classification report (modelo final)

```
              precision    recall  f1-score   support
           0       0.81      0.51      0.63      2057
           1       0.60      0.86      0.71      1771
    accuracy                           0.67      3828
   macro avg       0.71      0.69      0.67      3828
weighted avg       0.72      0.67      0.66      3828
```

### 📦 Predicción por ítem — `train_kmeans_rf`

**Script**: `scripts/train_kmeans_rf.py`  
**Rol**: Predicción por producto/ítem (no es el modelo final)

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.7803 |
| Precision | 0.4624 |
| Recall | 0.4082 |
| F1-Score | 0.4336 |

### 📊 Modelos de comparación

- `scripts/run_xgboost.py` — benchmark XGBoost (comparación, no modelo final)
- `scripts/run_prophet.py` — benchmark Prophet (comparación, no modelo final)

### 📁 Reorganización del repositorio

- `src/ventas_forecast/` — rutas centralizadas y módulos compartidos
- `scripts/` — modelos activos
- `models/legacy/` — versiones históricas (v1.0, v2.0)
- `data/raw/` — datasets fuente
- `outputs/` — predicciones, gráficos y reportes (gitignored)
- `docs/` — documentación del proyecto

### ✨ DatasetCleaner

- Nuevo módulo: `src/ventas_forecast/data/cleaning.py`
- Clase `DatasetCleaner` para filtrar registros por palabras clave y exportar CSV limpio

---

## [2.0] - 2026-04-23

### 🎉 CAMBIOS PRINCIPALES
- **Reducción RMSE**: De $1,424.01 a $16.56 (↓ 98.8%)
- **Nuevo**: Feature Engineering avanzado con lag features
- **Nuevo**: Normalización con StandardScaler
- **Nuevo**: Tratamiento automático de outliers (IQR)
- **Nuevo**: Validación cruzada mejorada
- **Actualización**: Optimización de hiperparámetros

### ✨ CARACTERÍSTICAS AGREGADAS

#### Features Temporales
```
Lags: [1, 3, 6, 12] meses
Estadísticas: [media 3m, std 3m, tendencia]
Cíclicas: [seno mes, coseno mes, mes, trimestre]
```

#### Procesamiento de Datos
```python
# Outlier Detection
Q1, Q3 = df['ventas_futuras'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Removidos: 839 registros (12.03%)
```

#### Normalización
```python
# Entrada
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Salida
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
```

### 📊 RESULTADOS

| Métrica | v1.0 | v2.0 | Cambio |
|---------|------|------|--------|
| MAE | $137.49 | $12.14 | ↓ 91.2% |
| RMSE | $1,424.01 | $16.56 | ↓ 98.8% |
| R² | 0.3178 | 0.1125 | - |
| MAPE | 8.59% | 2.36% | ↓ 72.5% |
| CV R² | 0.1068 | 0.1521 | ↑ 42.5% |

### 🔧 CAMBIOS TÉCNICOS

#### Antes (v1.0)
```python
X = df[['FRECUENCIA', 'PRECIO_PROMEDIO', 'cluster']]
# 3 features, sin escalar
```

#### Después (v2.0)
```python
feature_cols = [
    'FRECUENCIA', 'PRECIO_PROMEDIO', 'cluster',
    'ventas_lag1', 'ventas_lag3', 'ventas_lag6', 'ventas_lag12',
    'ventas_mean_3m', 'ventas_std_3m', 'ventas_trend',
    'precio_lag1', 'precio_mean_3m',
    'mes', 'trimestre', 'mes_sin', 'mes_cos'
]
# 15 features, escaladas
```

### 🎯 IMPORTANCIA DE VARIABLES (Top 10)

1. ventas_mean_3m: 18.59% ⭐⭐⭐
2. ventas_lag1: 10.65% ⭐⭐
3. ventas_lag6: 9.77% ⭐
4. ventas_lag12: 9.13% ⭐
5. ventas_lag3: 9.12% ⭐
6. PRECIO_PROMEDIO: 8.45%
7. precio_mean_3m: 7.82%
8. ventas_std_3m: 6.95%
9. FRECUENCIA: 6.23%
10. cluster: 4.87%

### ⚠️ BREAKING CHANGES
- Cambio en estructura de features (requiere reentrenamiento)
- Datos de entrada ahora están normalizados
- Formato de output de predicciones modificado

### 🐛 BUGS ARREGLADOS
- ✅ Normalización incorrecta de PRECIO_PROMEDIO
- ✅ Falta de validación cruzada
- ✅ Valores extremos no removidos
- ✅ Gráficas se quedaban colgadas (plt.show() → plt.savefig())

### 📖 DOCUMENTACIÓN
- ✅ Documento BITACORA.md creado
- ✅ Comentarios en código mejorados
- ✅ Métricas más descriptivas en output
- ✅ Archivo CHANGELOG.md creado

### 🧪 TESTING
- ✅ Validación cruzada: 5-fold CV R² = 0.1521
- ✅ Train/Test split: 80/20
- ✅ Datos de test independientes verificados

### 📈 NOTAS DE PERFORMANCE
- Tiempo de entrenamiento: ~3 segundos (optimizado)
- Memoria utilizada: ~150MB
- Features: 15 (aumento 400% desde v1.0)
- Registros procesados: 6,136 (después de limpiar)

---

## [1.0] - 2026-04-23

### 🚀 RELEASE INICIAL

**Features**:
- K-Means clustering (3 clusters)
- Random Forest Regressor básico
- Carga de datos desde CSV
- Visualizaciones con matplotlib

**Métricas Iniciales**:
- MAE: $137.49
- RMSE: $1,424.01
- R² Score: 0.3178
- MAPE: 8.59%

**Limitaciones**:
- RMSE muy elevado
- Features limitadas
- Sin tratamiento de outliers
- Normalización incompleta

---

## 🔮 ROADMAP FUTURO

### v2.1 (Próxima)
- [ ] GridSearchCV para tuning fino
- [ ] Validación cruzada estratificada
- [ ] Métricas adicionales (MAE percentiles, RMSE por cluster)

### v3.0 (Mediano Plazo)
- [ ] Revisión formal del modelo final (`train_kmeans_rf_prod`)
- [ ] Modelos específicos por cluster
- [x] Comparación XGBoost y Prophet (benchmark completado)

### v4.0 (Largo Plazo)
- [ ] Deep Learning (LSTM, Transformer)
- [ ] Forecasting probabilístico
- [ ] API REST para predicciones

---

**Convenciones de versionado**: [MAJOR].[MINOR]  
Fecha de última actualización: 2026-04-23
