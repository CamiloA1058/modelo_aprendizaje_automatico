# Datos históricos (V1 a V4)

Versiones anteriores del dataset de ventas. Se conservan como referencia y para
reproducir los scripts de `legacy/`. **El modelo de producción usa solo
`data/raw/Query_Result_V5.csv`.**

| Archivo | Registros | Origen | Qué cambia respecto al anterior |
|---------|-----------|--------|---------------------------------|
| `Query_Result.csv` (V1) | 41.031 | Consulta A de `sql/consultas_versiones_anteriores.sql` | Primera exportación: ventas mensuales por producto (`ANIO`, `MES`, `VENTAS_MENSUALES`, `FRECUENCIA`, `PRECIO_PROMEDIO`, `TOTAL_VENDIDO`). |
| `Query_Result_V2.csv` (V2) | 87.885 | Consulta B (misma colección de consultas) | Granularidad diaria; agrega `FECHA`, `DIA`, `DIA_SEMANA`, `VENTAS`, `HUBO_PROMOCION`, `TEMPORADA`. Sin filtros. |
| `Query_Result_V3.csv` (V3) | 86.536 | V2 limpiado fuera de la base de datos | Elimina 1.349 registros cuya descripción contiene `MANTENIMIENTO` o `SOLUCION` (mano de obra y gasolina). No hay consulta SQL de V3. |
| `Query_Result_V4.csv` (V4) | 78.945 | Consulta C (misma colección de consultas) | Excluye ítems con `VALORU = 1` (sin precio configurado) y grupos con precio promedio <= 10. |
| `Query_Result_V5.csv` (V5, en `data/raw/`) | 78.921 | `sql/extraccion_ventas_v5.sql` | Filtros adicionales contra datos corruptos (cantidades y totales absurdos, totales cero o negativos). |

Otros archivos:

- `ventas.csv`: serie sintética mínima que usa `legacy/models/arima_rf_demo.py`.
- `cleaner.py`: script original que limpiaba V4 con `DatasetCleaner`
  (`src/ventas_forecast/data/cleaning.py`). Solo se conserva como referencia.

Formato común de los CSV: separador `;`, números con coma decimal y punto de
miles (`1.234,50`), fechas `dd/mm/aaaa`, codificación UTF-8 con BOM.
