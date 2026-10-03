# SQL de extracción

Consultas contra la base de datos Firebird del sistema de inventario
(tablas `DOCINV`, `ITEMDOCINV`, `PRODUCTO`). Los CSV resultantes alimentan al modelo.

| Archivo | Contenido |
|---------|-----------|
| `extraccion_ventas_v5.sql` | Consulta vigente. Genera `data/raw/Query_Result_V5.csv`. |
| `consultas_versiones_anteriores.sql` | Consultas de V1, V2 y V4, más consultas exploratorias. |
| `exploracion/explorar_bd.py`, `explorar_bd_v2.py` | Scripts que ejecutan consultas de exploración y exportan CSV. Las credenciales se leen de variables de entorno. |
| `exploracion/.env.example` | Plantilla de variables de entorno (sin valores). |

## Linaje de los datasets

```
V1  consulta A (mensual por producto)
V2  consulta B (diaria, sin filtros)
V3  V2 sin descripciones MANTENIMIENTO / SOLUCION (paso externo, sin SQL)
V4  consulta C (diaria, VALORU > 1, HAVING AVG(VALORU) > 10)
V5  extraccion_ventas_v5.sql (filtros adicionales de cantidades y totales)
```

V1 a V4 están en `data/historico/`; V5 en `data/raw/`.

## Cómo obtener el CSV de producción

1. Abrir un cliente Firebird (por ejemplo IBExpert o FlameRobin) conectado a la base.
2. Ejecutar `extraccion_ventas_v5.sql`. Ajustar el rango de `d.FECHA` en la
   cláusula `WHERE` si se necesitan meses nuevos.
3. Exportar el resultado a CSV con separador `;`, números en formato regional
   (coma decimal, punto de miles) y fechas `dd/mm/aaaa`, con encabezados.
4. Guardar el archivo como `data/raw/Query_Result_V5.csv` (o subirlo desde el dashboard).

## Scripts de exploración

Requieren el driver `fdb` (`pip install fdb`) y estas variables de entorno:
`FIREBIRD_HOST`, `FIREBIRD_DATABASE`, `FIREBIRD_USER`, `FIREBIRD_PASSWORD`
(opcional: `FIREBIRD_OUTPUT_DIR`). Si falta alguna, el script termina con un
mensaje claro. Copiar `.env.example` a `.env` (ignorado por git) como referencia
local, y exportar las variables en la terminal antes de ejecutar:

```bash
python sql/exploracion/explorar_bd_v2.py
```

Nunca se versionan credenciales.
