# 📝 CHANGELOG - Control de Versiones

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
