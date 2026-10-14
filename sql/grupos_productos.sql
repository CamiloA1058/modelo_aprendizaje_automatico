-- Catalogo producto -> grupo (solo datos de catalogo, sin clientes).
-- Genera data/raw/grupos_productos.csv con sql/extraer_grupos.py.
SELECT TRIM(p.CODIGO) AS CODIGO,
       p.IDGRUPO      AS ID_GRUPO,
       TRIM(g.DESCRIPCION) AS GRUPO
FROM PRODUCTO p
LEFT JOIN GRUPO g ON g.IDGRUPO = p.IDGRUPO
ORDER BY p.CODIGO
