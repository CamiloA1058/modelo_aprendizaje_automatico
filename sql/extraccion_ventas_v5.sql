-- ============================================================
-- extraccion_ventas_v5.sql
-- Produce data/raw/Query_Result_V5.csv (dataset de produccion).
--
-- Versión final con filtros completos contra datos corruptos
--
-- Problemas que resuelve vs V4:
--   1. VALORU = 1  -> productos sin precio configurado
--   2. Overflow numérico en TOTALITEM -> valores de billones de pesos
--   3. Cantidades absurdas en CANTREAL -> 0.25 unidades de banda, etc.
--   4. Items con total cero o negativo
--
-- ============================================================

SELECT
    p.CODIGO,
    p.DESCRIPCION,
    d.FECHA,
    EXTRACT(YEAR  FROM d.FECHA) AS ANIO,
    EXTRACT(MONTH FROM d.FECHA) AS MES,
    EXTRACT(DAY   FROM d.FECHA) AS DIA,
    EXTRACT(WEEKDAY FROM d.FECHA) AS DIA_SEMANA,
    SUM(i.CANTREAL)             AS VENTAS,
    SUM(i.TOTALITEM)            AS TOTAL_VENDIDO,
    AVG(i.VALORU)               AS PRECIO_PROMEDIO,
    COUNT(DISTINCT d.IDDOCINV)  AS FRECUENCIA,

    -- PROMOCIONES
    CASE
        WHEN SUM(i.TOTALDESCUENTO) > 0 THEN 1
        ELSE 0
    END AS HUBO_PROMOCION,

    -- TEMPORADA
    CASE
        WHEN EXTRACT(MONTH FROM d.FECHA) IN (11, 12) THEN 'Alta'
        WHEN EXTRACT(MONTH FROM d.FECHA) IN (6, 7)   THEN 'Media'
        ELSE 'Baja'
    END AS TEMPORADA

FROM DOCINV d
INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO

WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'

  -- FILTRO 1: excluir ítems sin precio real configurado (VALORU = 1)
  AND i.VALORU > 10

  -- FILTRO 2: excluir cantidades absurdas
  -- Ningún producto debería venderse en más de 10.000 unidades en una sola factura
  AND i.CANTREAL > 0
  AND i.CANTREAL < 10000

  -- FILTRO 3: excluir ítems con total por factura absurdo
  -- Máximo 50 millones COP por ítem por factura (ajustar si hay productos costosos)
  AND i.TOTALITEM > 0
  AND i.TOTALITEM < 50000000

GROUP BY
    p.CODIGO,
    p.DESCRIPCION,
    d.FECHA

-- FILTRO 4: excluir grupos donde el precio promedio o total agregado
-- sigue siendo anómalo después de agrupar por día
HAVING AVG(i.VALORU) > 10
   AND AVG(i.VALORU) < 20000000
   AND SUM(i.TOTALITEM) > 0
   AND SUM(i.TOTALITEM) < 50000000

ORDER BY d.FECHA;
