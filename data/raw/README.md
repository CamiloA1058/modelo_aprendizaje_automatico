# Datos de entrada (producción)

Aquí vive únicamente el dataset que consume el modelo:

- `Query_Result_V5.csv`: ventas diarias por producto (enero de 2024 a abril
  de 2026), extraídas con [`sql/extraccion_ventas_v5.sql`](../../sql/extraccion_ventas_v5.sql).

Para regenerarlo o actualizarlo, ejecutar esa consulta en un cliente Firebird,
exportar con las opciones descritas en [`sql/README.md`](../../sql/README.md) y
reemplazar este archivo. Las versiones V1 a V4 están en `data/historico/`.

Columnas: `CODIGO;DESCRIPCION;FECHA;ANIO;MES;DIA;DIA_SEMANA;VENTAS;TOTAL_VENDIDO;PRECIO_PROMEDIO;FRECUENCIA;HUBO_PROMOCION;TEMPORADA`.
