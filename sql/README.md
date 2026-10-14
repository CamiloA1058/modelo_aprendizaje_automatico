# SQL de extracción

Consultas contra la base de datos Firebird del sistema de inventario
(tablas `DOCINV`, `ITEMDOCINV`, `PRODUCTO`). Los CSV resultantes alimentan al modelo.

| Archivo | Contenido |
|---------|-----------|
| `extraccion_ventas_v6.sql` | Consulta vigente. Igual que la V5 pero solo facturas de venta (tipos 1 y 13) no anuladas y sin servicios (mano de obra y fletes, lista explícita de códigos de producto). Genera `data/raw/Query_Result_V6.csv`. |
| `extraer_ventas.py` | Extractor offline: ejecuta un `.sql` sobre una copia de la base Firebird y escribe el CSV en el formato exacto de la V5 (`--db --sql --out --desde --hasta`). |
| `extraccion_ventas_v5.sql` | Consulta anterior (sin filtro de tipo de documento). Genera `data/historico/Query_Result_V5.csv`. |
| `consultas_versiones_anteriores.sql` | Consultas de V1, V2 y V4, más consultas exploratorias. |
| `exploracion/explorar_bd.py`, `explorar_bd_v2.py` | Scripts que ejecutan consultas de exploración y exportan CSV. Las credenciales se leen de variables de entorno. |
| `exploracion/.env.example` | Plantilla de variables de entorno (sin valores). |

## Linaje de los datasets

```
V1  consulta A (mensual por producto)
V2  consulta B (diaria, sin filtros)
V3  V2 sin descripciones MANTENIMIENTO / SOLUCION (paso externo, sin SQL)
V4  consulta C (diaria, VALORU > 1, HAVING AVG(VALORU) > 10)
V5  extraccion_ventas_v5.sql (filtros adicionales de cantidades y totales; mezcla tipos de documento)
V6  extraccion_ventas_v6.sql (V5 + solo facturas de venta tipos 1 y 13, no anuladas, sin servicios)
```

V1 a V5 están en `data/historico/`; V6 en `data/raw/`.

## Exclusión de servicios

La V6 excluye por código de producto los servicios (`1250` MANTENIMIENTO, `2660`
SERVICIO DE TORNO, `TORNO SPOOL`, `3087` y `FLETE`, `SEPILLADADECAMPANAS`): el
modelo predice demanda de repuestos y un servicio no tiene existencias. No se usa
`PRODUCTO.SERVICIO` porque marca repuestos reales y omite MANTENIMIENTO. Para
agregar un servicio nuevo, añadir su código al FILTRO 5 del `.sql` (la prueba
`tests/test_extraer_ventas.py` valida la lista). Detalle en `docs/CHANGELOG.md`.

## Cómo obtener el CSV de producción

Trabajar siempre sobre una COPIA de la base, nunca la original. Con el cliente
Firebird embebido (`fbclient.dll`, sin servidor) y el paquete `firebird-driver`
(`pip install firebird-driver`):

```bash
export FIREBIRD_CLIENT=/ruta/a/fbclient.dll   # usuario: FIREBIRD_USER (SYSDBA); clave opcional: FIREBIRD_PASSWORD
python sql/extraer_ventas.py --db copia.fdb --sql sql/extraccion_ventas_v6.sql     --out data/raw/Query_Result_V6.csv --desde 2024-01-01 --hasta 2026-03-27
```

El extractor reemplaza el rango de `d.FECHA` del `.sql` por `--desde/--hasta` y
escribe el CSV con separador `;`, UTF-8 con BOM, CRLF, fechas `dd/mm/aaaa` y
números con coma decimal y punto de miles, igual que la V5. Alternativa manual:
ejecutar la consulta en un cliente Firebird y exportar con ese mismo formato.
El CSV de ventas posteriores al entrenamiento (`data/holdout/`) se genera con la
misma consulta y otro rango de fechas.

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
