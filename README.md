# Modelo de Predicción de Ventas K-Means + Random Forest

Sistema de predicción de ventas con una acción por producto (Comprar más / Mantener / No reponer este mes / Revisar / Por promedio) utilizando K-Means + Random Forest.

**Modelo final (producción)**: `src/ventas_forecast/model.py` (clase `SalesForecastModel`), que se entrena con `scripts/entrenar.py` o desde el dashboard (`dashboard/app.py`).  
**Código heredado** (no es producción): `legacy/` (primeros pipelines, versiones 1.0 y 2.0, benchmarks sueltos de XGBoost/Prophet).  
**Modelos de comparación** (benchmark): `scripts/compare_models.py` (RF vs XGBoost vs Prophet vs ARIMA, sobre la misma partición del modelo final).

Dataset vigente: `data/raw/Query_Result_V6.csv` (facturas de venta reales del taller de los últimos dos años, sin servicios; 48.028 registros diarios, 3.584 productos).

Regla de validación: nada de lo reportado se elige con los datos sobre los que se reporta. Hiperparámetros, peso de la mezcla, umbral de venta, cuantiles del stock, umbral t de Comprar más, s_baja de No reponer, el monto si se vende y la regla de stock por acción se eligen solo con el entrenamiento (validación cruzada mensual o validación interna temporal); la prueba interna (último 20 % de los meses de cada producto) solo reporta y el estudio de caso es la evaluación final (entrada "[Validación] V1" de `docs/CHANGELOG.md`).

### Métricas — clasificador (prueba interna n=6.242, `scripts/entrenar.py`)

Clasificador con las variables de regularidad del modelo de venta y `class_weight="balanced_subsample"` (entrada "[Sistema] Acción única por producto" de `docs/CHANGELOG.md`; antes F1 macro 0,4502, exactitud balanceada 0,4974, exactitud 0,6831, según `outputs/reports/evaluacion_modelo.txt`, que no se regeneró). Métricas de la clase más probable, sin el umbral t ni la coherencia con la venta.

Métricas principales (robustas al desbalance de clases; Reducir = 79,80 % de la prueba):

| Métrica | Modelo | Línea base (siempre "Reducir") |
|---|---|---|
| F1 macro | 0,4620 | 0,2959 |
| Exactitud balanceada | 0,5104 | 0,3333 |

Métricas ponderadas por soporte de clase (referencia):

| Métrica | Modelo | Línea base (siempre "Reducir") |
|---|---|---|
| Exactitud | 0,7432 | 0,7980 |
| F1 ponderado | 0,7571 | 0,7083 |

### Métricas — regresor (`outputs/reports/evaluacion_modelo.txt`, prueba interna, top productos, n=4.472)

| Métrica | Modelo (RF) | Mezcla (w = 0,0) | Ingenuo | Media_3 |
|---|---|---|---|---|
| MAE (COP) | 232.349 | 314.974 | 427.404 | 314.974 |
| RMSE (COP) | 907.724 | 930.639 | 1.254.946 | 930.639 |
| R² | 0,4213 | 0,3917 | −0,1061 | 0,3917 |
| MASE | 1,0382 | 1,4074 | 1,9097 | 1,4074 |
| Razón de sumas | 0,5937 | 1,2919 | — | 1,2919 |

Con w elegido en la validación interna (w = 0,0) la mezcla de abril es la media de 3 meses.

Ver [docs/CHANGELOG.md](docs/CHANGELOG.md) para el historial completo de decisiones y resultados, y [docs/BITACORA.md](docs/BITACORA.md) para reportes detallados.

---

## Inicio rápido

**Windows (cliente final)**: doble clic en `Iniciar_Dashboard.bat`. Usa `.venv` si existe, instala las dependencias la primera vez y abre `http://localhost:5000`.

**Desarrollador**:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# Solo si se va a ejecutar scripts/compare_models.py:
pip install -r requirements-comparacion.txt
python dashboard/app.py          # dashboard en http://localhost:5000
python scripts/entrenar.py       # entrena y escribe outputs/predictions/PREDICCION_MENSUAL.csv
```

Usar `PYTHONIOENCODING=utf-8` para evitar errores de consola con los caracteres ✓/✅/❌. `scripts/entrenar.py` termina mostrando una figura con matplotlib; en entornos sin pantalla definir `MPLBACKEND=Agg`.

---

## Estructura del proyecto

```
modelo_aprendizaje_automatico/
├── README.md
├── requirements.txt             # Dependencias del modelo y el dashboard
├── requirements-comparacion.txt # Extra para compare_models.py (XGBoost, Prophet, ARIMA)
├── Iniciar_Dashboard.bat        # Lanzador Windows, doble clic
│
├── src/ventas_forecast/         # Código compartido
│   ├── model.py                 # Modelo FINAL: SalesForecastModel
│   ├── paths.py                 # Rutas centralizadas (DEFAULT_DATASET, UPLOADS, ...)
│   ├── eda.py                   # Funciones de análisis exploratorio
│   ├── estado_productos.py      # Estado de todos los productos (cinco estados)
│   ├── benchmarks.py            # Utilidades de comparación RF/XGBoost/Prophet/ARIMA
│   └── data/cleaning.py         # DatasetCleaner + remove_bulk_adjustments
│
├── dashboard/                   # Interfaz web
│   ├── app.py                   # Servidor Flask (API JSON)
│   └── dashboard.html           # Interfaz
│
├── scripts/                     # Scripts ejecutables
│   ├── entrenar.py              # Entrena el modelo final y exporta la predicción
│   ├── select_k.py              # Selección de k (codo + silueta) para K-means
│   ├── tune_rf.py               # Ajuste de hiperparámetros del clasificador
│   ├── evaluate_model.py        # Evaluación del clasificador y el regresor
│   ├── compare_models.py        # Comparación RF vs XGBoost vs Prophet vs ARIMA
│   └── eda.py                   # Reporte y figuras de análisis exploratorio
│
├── sql/                         # Extracción desde Firebird
│   ├── extraccion_ventas_v6.sql # Consulta vigente (genera Query_Result_V6.csv)
│   ├── extraer_ventas.py        # Extractor offline a CSV (formato V5)
│   ├── consultas_versiones_anteriores.sql
│   └── exploracion/             # Scripts de exploración (credenciales por variables de entorno)
│
├── data/
│   ├── raw/                     # Query_Result_V6.csv (producción)
│   ├── holdout/                 # Ventas posteriores al entrenamiento (estudio de caso)
│   └── historico/               # V1 a V5 y utilidades antiguas
│
├── tests/                       # Pruebas (unittest)
├── docs/                        # BITACORA.md (histórico) y CHANGELOG.md
├── legacy/                      # Código heredado, fuera de producción
└── outputs/                     # Generado al ejecutar (gitignored): figures/, predictions/, reports/, uploads/
```

---

## Cómo reentrenar con datos nuevos

1. Ejecutar `sql/extraer_ventas.py` con `sql/extraccion_ventas_v6.sql` sobre una copia de la base Firebird (instrucciones en [sql/README.md](sql/README.md)).
2. Guardar el resultado como `data/raw/Query_Result_V6.csv`, o subirlo desde la pestaña "Ejecutar modelo" del dashboard (se guarda en `outputs/uploads/`).
3. Entrenar con `python scripts/entrenar.py` o con el botón de ejecución del dashboard.
4. La predicción queda en `outputs/predictions/PREDICCION_MENSUAL.csv`.

---

## Cómo ejecutar los scripts

Desde la **raíz del proyecto**:

```bash
PYTHONIOENCODING=utf-8 python scripts/entrenar.py        # Modelo final
PYTHONIOENCODING=utf-8 python scripts/select_k.py        # Selección de k
PYTHONIOENCODING=utf-8 python scripts/tune_rf.py         # Hiperparámetros
PYTHONIOENCODING=utf-8 python scripts/evaluate_model.py  # Evaluación
PYTHONIOENCODING=utf-8 python scripts/compare_models.py  # Comparación de modelos
PYTHONIOENCODING=utf-8 python scripts/eda.py             # EDA
PYTHONIOENCODING=utf-8 python scripts/reporte_prediccion_vs_real.py  # Excel predicción vs real por producto (requiere openpyxl)
```

### Pruebas

```bash
PYTHONIOENCODING=utf-8 python -m unittest discover -s tests
```

Con `VENTAS_PRUEBA_DATOS_REALES=1` se ejecuta además una prueba que corre el modelo con `data/raw/Query_Result_V6.csv` (unos 30 s) y comprueba los conteos por estado de abril de 2026.

### Salidas

Los resultados se guardan en `outputs/` (gitignored):

- `outputs/predictions/` — CSV de predicciones (decisiones, probabilidades, ventas predichas)
- `outputs/figures/` — gráficos PNG (selección de k, matriz de confusión, real vs. predicho, comparación de modelos, EDA)
- `outputs/reports/` — resúmenes TXT/CSV (evaluación, tuning, comparación, EDA)
- `outputs/uploads/` — CSV subidos desde el dashboard

---

## Dashboard

Interfaz web local para usuarios sin conocimientos técnicos. Pestañas: resumen general (KPIs por acción, venta esperada, historial, distribución de acciones, productos para comprar más y para no reponer), vista por grupo de producto, estado de todos los productos, catálogo de predicciones con búsqueda, filtros y descarga del CSV, detalle por producto y ejecución del modelo con subida de CSV o lectura directa de la base de datos del taller (ver [Leer directamente de la base de datos](#leer-directamente-de-la-base-de-datos)) y parámetros.

**Acción única.** Cada producto tiene una sola acción, un monto y un stock coherentes entre sí (`SalesForecastModel.agregar_accion()`, activo por defecto, `accion_unica=False` lo desactiva; reglas en `src/ventas_forecast/accion.py`):

| Acción | Cuándo | Monto que se muestra | Stock sugerido |
|---|---|---|---|
| Comprar más | El clasificador da Reforzar con P(Reforzar) ≥ t y la probabilidad de venta alcanza el umbral | Venta esperada (prob. × monto si se vende) | 70 % |
| Mantener | Probabilidad de venta ≥ umbral con Mantener o Reducir | Venta esperada | 70 % |
| No reponer este mes | Probabilidad < s_baja, o Reducir con probabilidad < umbral | "Sin venta esperada" | 0 |
| Revisar | El resto (probabilidad entre s_baja y el umbral) | "Entre $0 y $X si se vende" | El monto predicho |
| Por promedio | No vendió el mes pasado (estimación por promedio) | Venta esperada, rango o "Venta poco probable" según la probabilidad | 70 % |

t, s_baja, el "monto si se vende" (media de 3 o 6 meses con venta o mediana de 6) y la regla de stock se eligen con la validación interna; la regla de stock por acción se adopta solo si allí mantiene la cobertura a no más de 3 puntos de la del 70 % para todos y baja el stock (en los seis meses del estudio de caso se adoptó). El total del resumen es la venta esperada de todos los productos con monto. En el estudio de caso (abril a septiembre de 2026): Comprar más acierta el 64,0 % (517 productos; antes Reforzar 41,4 % de 2.415), ninguno con "Se venderá: No" (antes 393), No reponer acierta el 79,7 %, 29 incoherencias de monto (antes 3.397), total 1,08 veces lo real, cobertura del stock 69,6 % con stock 1,35 veces lo vendido. Las tablas muestran la acción, el monto, el stock y la probabilidad de venta como barra; Prob. crecimiento, variación y segmento están en el detalle del producto.

**Predicción combinada.** El monto predicho por producto (`ventas_predichas`) es `w x RF + (1 - w) x media_3`, donde `media_3` es el promedio de los últimos 3 meses del producto y `w` (0, 0,1 ... 1) se elige con la validación interna dentro del entrenamiento (un regresor temporal ajustado con los primeros meses de entrenamiento de cada producto predice sus últimos meses de entrenamiento), minimizando |suma predicha - suma real|; la prueba interna solo reporta. El RF puro se conserva en `ventas_predichas_rf`. La mezcla sigue siendo la base del stock sugerido (el monto que se muestra es la venta esperada, ver "Acción única"); el CSV descargable trae ambas columnas. En la última corrida real `w` = 0,0 (en la validación interna el RF predice el 23,6 % de la suma real y la media de 3 meses el 94,8 %), así que el monto combinado de abril es la media de 3 meses; en la prueba interna la razón suma predicha/real es 0,59 (RF) y 1,29 (mezcla) (ver `outputs/reports/evaluacion_modelo.txt` y la entrada "[Validación] V1" de `docs/CHANGELOG.md`).

**Estimación por promedio.** Los productos que no vendieron en el último mes pero sí en alguno de los 2 meses anteriores (con al menos `min_months` meses con ventas y dentro de los 1.900 con más ventas) aparecen aparte: su monto es el promedio de los últimos 3 meses (`media_3`), no una predicción del modelo, y su decisión es "Sin recomendación (estimación por promedio)", porque el clasificador es peor que la clase mayoritaria en ellos. Se calculan en `SalesForecastModel.estimar_por_promedio()` (activo por defecto, `estimacion_por_promedio=False` lo desactiva) y el CSV trae la columna `origen_estimacion` ("modelo" / "promedio"). Su acción es "Por promedio" (tarjeta y filtro propios) y la venta esperada del resumen suma ambos grupos. En abril de 2026 (V6): 697 productos con decisión, 348 por promedio, 302,8 M + 60,6 M = 363,4 M (entrada "[Sistema] Estimación por promedio..." de `docs/CHANGELOG.md`).

**Probabilidad de venta.** Para todos los productos (los del modelo y los de estimación por promedio) el sistema calcula la probabilidad de que tengan ventas el mes predicho y la traduce a "Se venderá: Sí/No". `SalesForecastModel.agregar_probabilidad_venta()` (activo por defecto, `probabilidad_venta=False` lo desactiva) entrena un Random Forest binario solo con la partición de entrenamiento y elige el umbral con la validación interna dentro de ese entrenamiento (la exactitud que muestra el dashboard es la de la prueba interna, 79,2 % en abril); el CSV trae las columnas `prob_venta` (0 a 1) y `se_vendera`. La probabilidad de venta decide la acción (Comprar más exige "Se venderá: Sí"); las tablas la muestran como barra, el catálogo permite ordenar por ella, la tarjeta de productos dice cuántos probablemente se venderán y el detalle la muestra también para los de promedio. En el estudio de caso (abril a septiembre de 2026) acierta si habrá venta en 70,1 % de los producto-mes agrupados (69,0 % a 71,6 % por mes); los montos no cambian (entrada "[Sistema] Probabilidad de venta en el dashboard" de `docs/CHANGELOG.md`).

**Stock sugerido.** Para cada producto con monto predicho el sistema calcula un nivel de abastecimiento al 70 % y luego lo ajusta según la acción (No reponer 0, Revisar el monto predicho; ver "Acción única"). Con esa regla, en los meses de prueba (abril a septiembre de 2026) se cubrió la demanda de cerca de 7 de cada 10 productos (69,6 % agrupado, incluye los que no vendieron; stock total 1,35 veces lo vendido; con el 70 % para todos era 71,8 % y 1,45). `SalesForecastModel.agregar_stock_sugerido()` (activo por defecto, `stock_sugerido=False` lo desactiva, `nivel_servicio=0.70`) corrige la predicción con el cuantil 70 % de los errores de la validación interna (dentro del entrenamiento) por cubeta de probabilidad de venta; el CSV trae `stock_sugerido` (COP) y `stock_sugerido_unidades`, y el dashboard la columna "Stock sugerido" en Productos, detalle y Grupos. No cambia decisiones, montos ni totales (entrada "[Sistema] Stock sugerido por producto" de `docs/CHANGELOG.md`; la hoja de producto de `prediccion_vs_real.xlsx` trae "¿Cubrió la demanda?").

**Vista por grupo de producto.** La vista "Grupos" suma la venta esperada de los productos de cada grupo del catálogo `data/raw/grupos_productos.csv` (la línea o proveedor con la que compra el taller): productos con monto, cuántos hay en cada acción, venta esperada del grupo, ventas del mes anterior, variación y porcentaje del total, con buscador; un clic en un grupo abre "Productos" filtrado por él. `SalesForecastModel.agregar_resumen_grupo()` (activo por defecto, `resumen_por_grupo=False` lo desactiva) solo lee los resultados y no cambia ninguna predicción, decisión ni total. En abril a septiembre de 2026 esta suma acertó el 69,0 % del valor vendido por grupo (74,1 % en los 20 grupos que más vendieron el mes anterior; con la predicción combinada era 68,9 % y 74,1 %); úsela como guía para planear pedidos por línea (entrada "[Sistema] Vista por grupo de producto" de `docs/CHANGELOG.md`). La hoja "Grupos" de `prediccion_vs_real.xlsx` trae la comparación contra lo real mes a mes.

**Estado de todos los productos.** La vista "Todos los productos" muestra qué pasa con cada producto vendido en los datos, no solo con los que reciben predicción. Cada producto queda en uno solo de seis estados:

| Estado | Regla | Abril de 2026 (V6) |
|---|---|---|
| Con recomendación y monto | Vendió el último mes, tiene `min_months` (6) o más meses con ventas y está entre los 1.900 con más ventas | 645 |
| Solo recomendación | Igual, pero fuera de los 1.900: tiene decisión del modelo, sin monto | 52 |
| Monto por promedio | Entre los 1.900, 6 o más meses con ventas, no vendió el último mes y sí en los últimos 3 meses (estimación por promedio) | 348 |
| Producto nuevo | Menos de 6 meses con ventas y la primera venta en los últimos 6 meses | 441 |
| Venta ocasional | Menos de 6 meses con ventas, con la primera venta antes | 1.819 |
| Sin ventas recientes | 6 o más meses con ventas, pero sin una venta reciente que lo deje en los estados anteriores | 279 |

En total 3.584 productos, 1.045 con predicción. Arriba hay una tarjeta por estado con el número de productos, su porcentaje y su parte de las ventas de los últimos 12 meses, más una explicación corta para el cliente; un clic en la tarjeta filtra la tabla. La tabla (código, descripción, grupo, estado, meses con venta, última venta, ventas de 12 meses y, si tiene predicción, acción y monto) permite buscar, filtrar por estado, ordenar por ventas de 12 meses, meses sin vender o código, abrir el detalle de los productos con predicción y descargar todo en CSV (separador `;`, coma decimal). `SalesForecastModel.clasificar_estado_productos()` (lo llama `run()` al final y también el dashboard) deja la tabla en `model.estado_productos` y no cambia ningún cálculo ni columna existente. Los nombres y las explicaciones de los estados están solo en `ESTADOS` de `src/ventas_forecast/estado_productos.py`, para poder cambiarlos sin tocar la vista (entrada "[Sistema] Estado de todos los productos" de `docs/CHANGELOG.md`).

```
Navegador (dashboard/dashboard.html)
        │  HTTP / JSON
        ▼
  dashboard/app.py (Flask, puerto 5000)
        ├── GET  /api/status        Estado del pipeline + mes predicho + fuente (csv/bd) y rango leído
        ├── POST /api/run           Lanza el modelo en un hilo secundario (fuente=csv | bd)
        ├── GET  /api/bd/estado     ¿Está configurada la lectura de la base? (sin rutas ni claves)
        ├── GET  /api/summary       KPIs, conteo por acción (`acciones`, `acciones_catalogo`), pred_total (venta esperada: modelo + promedio), pred_total_modelo, pred_total_promedio, n_promedio, n_se_vendera, sale_threshold, umbral_reforzar, s_baja, regla_stock, historial, métricas, Comprar más / No reponer, conteo por estado (`estados`)
        ├── GET  /api/products      Lista paginada con filtros (acción, grupo) y orden
        ├── GET  /api/grupos        Venta esperada y productos por acción por grupo de producto
        ├── GET  /api/estados       Productos y ventas de 12 meses por estado (todos los productos)
        ├── GET  /api/estados/productos   Lista paginada de todos los productos (estado, búsqueda, grupo, orden)
        ├── GET  /api/estados/descargar   CSV con el estado de todos los productos
        ├── GET  /api/product/<id>  Historial + predicción de un producto
        └── GET  /api/download      Descarga PREDICCION_MENSUAL.csv
              ▼
  SalesForecastModel (src/ventas_forecast/model.py)
```

## Leer directamente de la base de datos

En **Ejecutar modelo → Origen de los datos** se elige entre **Subir archivo CSV** (como siempre) y **Leer de la base de datos**. Con la base, el dashboard lee las ventas con la misma consulta V6 del CSV (`sql/extraccion_ventas_v6.sql`: facturas de venta tipos 1 y 13 no anuladas, sin servicios) y el catálogo de grupos, y corre el mismo pipeline. Con el mismo rango de fechas el resultado es idéntico al CSV: contra una copia de la base, 2024-01-01 a 2026-03-31 da las mismas 48.028 filas, valores y tipos que `data/raw/Query_Result_V6.csv`.

**Qué configurar.** Copiar `config/base_datos.ejemplo.ini` como `config/base_datos.ini` (git lo ignora) y completar la sección `[firebird]`, o definir las variables de entorno, que tienen prioridad:

| Clave del archivo | Variable de entorno | Contenido |
|---|---|---|
| `base_datos` | `FIREBIRD_DB` | Ruta del `.fdb` (en este equipo, o en el servidor si hay `host`) — obligatoria |
| `host` | `FIREBIRD_HOST` | Vacío: se abre el archivo en este equipo. Con servidor Firebird: nombre o IP del servidor |
| `usuario` | `FIREBIRD_USER` | Usuario de Firebird (por defecto `SYSDBA`) |
| `clave` | `FIREBIRD_PASSWORD` | Clave del usuario |
| `cliente` | `FIREBIRD_CLIENT` | Ruta de `fbclient.dll` — obligatoria |
| `meses` | `FIREBIRD_MESES` | Meses completos que se leen (por defecto 24) |

**De dónde sale `fbclient.dll`.** Del Firebird instalado en el taller (servidor o cliente), por ejemplo `C:\Program Files\Firebird\Firebird_5_0\fbclient.dll`. El paquete de Python `firebird-driver` (en `requirements.txt`) necesita un cliente de Firebird 3 o posterior; un cliente nuevo puede conectarse por red a un servidor más antiguo.

**Solo lectura.** Cada lectura abre una transacción SNAPSHOT de solo lectura (el motor rechaza cualquier escritura y el sistema verifica que la transacción quedó en ese modo antes de leer), cada sentencia se valida como un único `SELECT` y la transacción termina con *rollback*. Las claves no aparecen en el registro, en las respuestas de la API ni en la interfaz.

**Qué meses se leen.** Se toma la fecha de la última venta válida. Si no es el último día de su mes, ese mes está incompleto y no se usa: se leen los `meses` meses completos anteriores y se predice justamente ese mes. Ejemplo: última venta 22/09/2026 → se leen del 01/09/2024 al 31/08/2026 y se predice septiembre de 2026. Si la última venta es el último día del mes, ese mes entra y se predice el siguiente. El registro de ejecución muestra los registros y el rango leídos, y la pantalla el rango y el mes predicho.

**Problemas frecuentes.**

| Mensaje | Qué revisar |
|---|---|
| "la lectura de la base de datos no está configurada: falta …" | Que exista `config/base_datos.ini` con `base_datos` y `cliente` (o las variables `FIREBIRD_*`) |
| "no se encontró el cliente de Firebird (fbclient.dll)" / "no se pudo cargar el cliente" | La ruta `cliente`, y que el `fbclient.dll` sea de 64 bits como Python y de Firebird 3 o posterior |
| "no se encontró el archivo de la base de datos" | La ruta `base_datos` (sin `host` debe existir en este equipo) |
| "no se pudo conectar a la base de datos" | Usuario y clave; si el servidor Firebird tiene la base abierta, configurar `host = localhost` (o la IP del servidor) para conectarse a través de él |
| "no se pudo leer la base de datos" | Que sea la base de ventas del taller y que el usuario tenga permiso de lectura |
| "falta el paquete firebird-driver" | Ejecutar `python -m pip install -r requirements.txt` (o reiniciar `Iniciar_Dashboard.bat`, que lo instala) |

En todos los casos la opción **Subir archivo CSV** sigue funcionando. Guía paso a paso para el técnico del taller: [docs/CONFIGURAR_BASE_DE_DATOS.md](docs/CONFIGURAR_BASE_DE_DATOS.md).

Pruebas: `tests/test_fuente_bd.py` y `tests/test_dashboard_fuente_bd.py` usan una conexión simulada. La prueba de integración contra una **copia** de la base se activa con `FIREBIRD_DB_PRUEBA` (ruta de la copia) y `FIREBIRD_CLIENT_PRUEBA` (ruta de `fbclient.dll`); sin ellas se omite.

## Uso programático del modelo

```python
import sys
sys.path.insert(0, "src")
from ventas_forecast.model import SalesForecastModel
from ventas_forecast.paths import DEFAULT_DATASET

model = SalesForecastModel(filepath=DEFAULT_DATASET, min_months=6)
model.run()                 # o paso a paso: load_and_clean, build_features, cluster, ...
print(model.results.head())
```

Si no se indica `output_path`, la predicción se escribe en `outputs/predictions/PREDICCION_MENSUAL.csv`. Los parámetros del constructor (`n_clusters`, `lags`, `growth_threshold`, `clf_threshold`, `top_n_products`, `col_map`, ...) están documentados en el docstring de la clase.

---

## Datos

| Ubicación | Uso |
|-----------|-----|
| `data/raw/Query_Result_V6.csv` | **Producción**: el modelo, los scripts y el dashboard |
| `data/historico/` | V1 a V4, solo referencia |

Ver [data/historico/README.md](data/historico/README.md) para el linaje V1 a V5 y [sql/README.md](sql/README.md) para la extracción. El detalle de la limpieza aplicada sobre V5 (completado de calendario mensual y exclusión de ajustes masivos de inventario) está en [docs/CHANGELOG.md](docs/CHANGELOG.md).

---

## Documentación

- [docs/CHANGELOG.md](docs/CHANGELOG.md) — Historial de decisiones y resultados (incluye la tabla de rutas antiguas y nuevas de la reestructuración)
- [docs/BITACORA.md](docs/BITACORA.md) — Bitácora histórica de desarrollo
- [sql/README.md](sql/README.md) — Extracción de datos
- [docs/CONFIGURAR_BASE_DE_DATOS.md](docs/CONFIGURAR_BASE_DE_DATOS.md) — Configurar la lectura directa de la base de datos en el taller

## Dependencias

Ver [requirements.txt](requirements.txt) y, para la comparación de modelos, [requirements-comparacion.txt](requirements-comparacion.txt).
