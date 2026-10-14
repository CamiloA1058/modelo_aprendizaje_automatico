# Datos de entrada (producción)

Aquí vive únicamente el dataset que consume el modelo:

- `Query_Result_V6.csv`: ventas diarias por producto (2024-01-09 a 2026-03-27),
  solo facturas de venta no anuladas (tipos 1 y 13), extraídas con
  [`sql/extraccion_ventas_v6.sql`](../../sql/extraccion_ventas_v6.sql).

Para regenerarlo, usar el extractor offline `sql/extraer_ventas.py` sobre una
copia de la base (ver [`sql/README.md`](../../sql/README.md)). La V5 (sin filtro
de tipo de documento) y las versiones V1 a V4 están en `data/historico/`.
Las ventas posteriores al entrenamiento, para el estudio de caso, están en
`data/holdout/`.

Columnas: `CODIGO;DESCRIPCION;FECHA;ANIO;MES;DIA;DIA_SEMANA;VENTAS;TOTAL_VENDIDO;PRECIO_PROMEDIO;FRECUENCIA;HUBO_PROMOCION;TEMPORADA`.
