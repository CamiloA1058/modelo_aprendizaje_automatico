-- ============================================================
-- consultas_versiones_anteriores.sql
-- Consultas que produjeron los datasets historicos (data/historico/)
-- y consultas exploratorias. La consulta de produccion vigente es
-- sql/extraccion_ventas_v5.sql.
--
-- Linaje de los CSV:
--   V1 -> consulta A (mensual por producto)
--   V2 -> consulta B (diaria por producto, sin filtros)
--   V3 -> V2 sin las filas cuya descripcion contiene MANTENIMIENTO o
--         SOLUCION (mano de obra y gasolina). Paso externo a la base de
--         datos: no hay consulta SQL de V3 (1.349 registros eliminados).
--   V4 -> consulta C (diaria, VALORU > 1, HAVING AVG(VALORU) > 10)
--   V5 -> sql/extraccion_ventas_v5.sql
-- ============================================================


-- ============================================================
-- A) Query_Result.csv (V1): ventas mensuales por producto
-- ============================================================
SELECT 
    p.CODIGO,
    p.DESCRIPCION,

    EXTRACT(YEAR FROM d.FECHA) AS anio,
    EXTRACT(MONTH FROM d.FECHA) AS mes,
    
    SUM(i.CANTREAL) AS ventas_mensuales,
    
    COUNT(DISTINCT d.IDDOCINV) AS frecuencia,  -- mejorado
    
    AVG(i.VALORU) AS precio_promedio,
    
    SUM(i.TOTALITEM) AS total_vendido  -- agregado

FROM DOCINV d

INNER JOIN ITEMDOCINV i
    ON i.IDDOCINV = d.IDDOCINV

INNER JOIN PRODUCTO p
    ON p.IDPRODUCTO = i.IDPRODUCTO

WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'

GROUP BY 
    p.CODIGO,
    p.DESCRIPCION,
    EXTRACT(YEAR FROM d.FECHA),
    EXTRACT(MONTH FROM d.FECHA)

ORDER BY 
    p.CODIGO,
    anio,
    mes;


-- ============================================================
-- B) Query_Result_V2.csv (V2): ventas diarias por producto, sin filtros
-- (V3 se obtuvo despues quitando MANTENIMIENTO y SOLUCION fuera de la BD)
-- ============================================================
SELECT 
    p.CODIGO,
    p.DESCRIPCION,

    d.FECHA,
    EXTRACT(YEAR FROM d.FECHA) AS anio,
    EXTRACT(MONTH FROM d.FECHA) AS mes,
    EXTRACT(DAY FROM d.FECHA) AS dia,
    EXTRACT(WEEKDAY FROM d.FECHA) AS dia_semana,

    SUM(i.CANTREAL) AS ventas,
    SUM(i.TOTALITEM) AS total_vendido,

    AVG(i.VALORU) AS precio_promedio,

    COUNT(DISTINCT d.IDDOCINV) AS frecuencia,

    --PROMOCIONES
    CASE 
        WHEN SUM(i.TOTALDESCUENTO) > 0 THEN 1
        ELSE 0
    END AS hubo_promocion,

    --TEMPORADA
    CASE 
        WHEN EXTRACT(MONTH FROM d.FECHA) IN (11,12) THEN 'Alta'
        WHEN EXTRACT(MONTH FROM d.FECHA) IN (6,7) THEN 'Media'
        ELSE 'Baja'
    END AS temporada

FROM DOCINV d

INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
INNER JOIN PRODUCTO p ON p.IDPRODUCTO = i.IDPRODUCTO

WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'

GROUP BY 
    p.CODIGO,
    p.DESCRIPCION,
    d.FECHA

ORDER BY d.FECHA;


-- ============================================================
-- C) Query_Result_V4.csv (V4): diaria, excluye items sin precio real
-- VALORU = 1 significa item registrado sin precio unitario
-- (cortesias, errores de digitacion, productos sin configurar)
-- ============================================================
SELECT 
    p.CODIGO,
    p.DESCRIPCION,
    d.FECHA,
    EXTRACT(YEAR FROM d.FECHA)    AS anio,
    EXTRACT(MONTH FROM d.FECHA)   AS mes,
    EXTRACT(DAY FROM d.FECHA)     AS dia,
    EXTRACT(WEEKDAY FROM d.FECHA) AS dia_semana,
    SUM(i.CANTREAL)               AS ventas,
    SUM(i.TOTALITEM)              AS total_vendido,
    AVG(i.VALORU)                 AS precio_promedio,
    COUNT(DISTINCT d.IDDOCINV)    AS frecuencia,

    -- PROMOCIONES
    CASE 
        WHEN SUM(i.TOTALDESCUENTO) > 0 THEN 1
        ELSE 0
    END AS hubo_promocion,

    -- TEMPORADA
    CASE 
        WHEN EXTRACT(MONTH FROM d.FECHA) IN (11, 12) THEN 'Alta'
        WHEN EXTRACT(MONTH FROM d.FECHA) IN (6, 7)   THEN 'Media'
        ELSE 'Baja'
    END AS temporada

FROM DOCINV d
INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
INNER JOIN PRODUCTO   p ON p.IDPRODUCTO = i.IDPRODUCTO

WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'

  -- FILTRO NUEVO: excluir ítems sin precio real configurado
  -- VALORU = 1 significa que el ítem fue registrado sin precio unitario
  -- (cortesías, errores de digitación, productos sin configurar)
  AND i.VALORU > 1

GROUP BY 
    p.CODIGO,
    p.DESCRIPCION,
    d.FECHA

-- Opcional: excluir grupos donde el precio promedio resultante
-- sigue siendo sospechosamente bajo (< 10 pesos)
HAVING AVG(i.VALORU) > 10

ORDER BY d.FECHA;


-- ============================================================
-- CONSULTAS EXPLORATORIAS (no producen un dataset del modelo)
-- Origen: consultas_sql.txt. Detalle por linea con cliente, conteo de
-- registros y agregado mensual temprano (con JOIN a TERCERO).
-- ============================================================

SELECT
    t.NOMBRE AS cliente,
    d.IDDOCINV,
    d.NUMDOC,
    d.FECHA,

    p.CODIGO,
    p.DESCRIPCION AS descripcion,

    i.CANTREAL AS cantidad,
    i.VALORU AS valor_unitario,

    i.TOTALCOSTONETO AS subtotal,
    i.TOTALIVA,
    i.TOTALITEM AS total_item,

    d.TOTALDOCUMENTO,

    CAST(d.MEMOOBSERVACIONES AS VARCHAR(500)) AS observaciones

FROM DOCINV d

INNER JOIN ITEMDOCINV i
    ON i.IDDOCINV = d.IDDOCINV

INNER JOIN PRODUCTO p
    ON p.IDPRODUCTO = i.IDPRODUCTO

INNER JOIN TERCERO t
    ON t.IDTERCERO = d.IDTERCERO
WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'
ORDER BY t.NOMBRE, d.IDDOCINV;



SELECT COUNT(*) AS total_registros
FROM (
    SELECT
        t.NOMBRE AS cliente,
        d.IDDOCINV,
        d.NUMDOC,
        d.FECHA,
        p.CODIGO,
        p.DESCRIPCION AS descripcion,
        i.CANTREAL AS cantidad,
        i.VALORU AS valor_unitario,
        i.TOTALCOSTONETO AS subtotal,
        i.TOTALIVA,
        i.TOTALITEM AS total_item,
        d.TOTALDOCUMENTO,
        CAST(d.MEMOOBSERVACIONES AS VARCHAR(500)) AS observaciones
    FROM DOCINV d
    INNER JOIN ITEMDOCINV i ON i.IDDOCINV = d.IDDOCINV
    INNER JOIN PRODUCTO p ON p.IDPRODUCTO = i.IDPRODUCTO
    INNER JOIN TERCERO t ON t.IDTERCERO = d.IDTERCERO
    WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'
) AS consulta;


SELECT 
    p.CODIGO,
    p.DESCRIPCION,

    EXTRACT(YEAR FROM d.FECHA) AS anio,
    EXTRACT(MONTH FROM d.FECHA) AS mes,
    
    SUM(i.CANTREAL) AS ventas_mensuales,
    COUNT(*) AS frecuencia,
    AVG(i.VALORU) AS precio_promedio

FROM DOCINV d

INNER JOIN ITEMDOCINV i
    ON i.IDDOCINV = d.IDDOCINV

INNER JOIN PRODUCTO p
    ON p.IDPRODUCTO = i.IDPRODUCTO

INNER JOIN TERCERO t
    ON t.IDTERCERO = d.IDTERCERO

WHERE d.FECHA BETWEEN '2024-01-01' AND '2026-04-13'

GROUP BY 
    p.CODIGO,
    p.DESCRIPCION,
    EXTRACT(YEAR FROM d.FECHA),
    EXTRACT(MONTH FROM d.FECHA)

ORDER BY 
    p.CODIGO,
    anio,
    mes;
