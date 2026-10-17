# Configurar la lectura de la base de datos

Guía para el técnico del taller. Al terminar, el dashboard podrá generar la predicción leyendo directamente la base de ventas, sin exportar un CSV.

**El sistema solo lee la base.** Usa una transacción de solo lectura y únicamente consultas `SELECT`; no escribe ni modifica datos.

## Antes de empezar

Necesita saber:

1. **Dónde está la base de ventas** (archivo `.fdb`) y si la usa un **servidor Firebird** (lo normal cuando el programa de facturación está abierto).
2. **Usuario y clave de Firebird** con permiso de lectura (por ejemplo `SYSDBA` y su clave).
3. **Dónde está `fbclient.dll`**: viene con el Firebird instalado en el taller, por ejemplo `C:\Program Files\Firebird\Firebird_5_0\fbclient.dll`. Debe ser de Firebird 3 o posterior y de 64 bits.

## Pasos

1. En la carpeta del programa, copie `config\base_datos.ejemplo.ini` y renombre la copia a `config\base_datos.ini`.
2. Abra `config\base_datos.ini` con el Bloc de notas y complete la sección `[firebird]`:

   | Clave | Qué escribir |
   |---|---|
   | `base_datos` | Ruta del archivo `.fdb` |
   | `host` | Vacío si la base está en este equipo y ningún servidor la tiene abierta. Si hay un servidor Firebird: `localhost` (mismo equipo) o el nombre o la IP del servidor. En ese caso `base_datos` es la ruta **en el servidor** |
   | `usuario` | Usuario de Firebird |
   | `clave` | Clave de ese usuario |
   | `cliente` | Ruta de `fbclient.dll` |
   | `meses` | Meses completos que se leen. Déjelo en `24` |

3. Guarde el archivo. No lo comparta: contiene la clave.
4. Abra el programa con `Iniciar_Dashboard.bat`. La primera vez instala lo que falte, incluido el paquete `firebird-driver`.
5. Vaya a **Ejecutar modelo**, elija **Leer de la base de datos** y verifique que aparezca "Base de datos configurada".
6. Pulse **Ejecutar predicción**. El registro debe mostrar una línea como "Leyendo la base de datos… 47.336 registros del 01/09/2024 al 31/08/2026", y al terminar la pantalla muestra el rango de datos usado y el mes predicho.

## Qué meses se usan

El sistema busca la fecha de la última venta. Si ese mes todavía no ha terminado, no lo usa: lee los 24 meses completos anteriores y predice ese mes.

**Ejemplo:** si la última venta es del 22/09/2026, lee del 01/09/2024 al 31/08/2026 y predice septiembre de 2026.

## Si algo falla

El mensaje en pantalla dice qué revisar. En todos los casos la opción **Subir archivo CSV** sigue funcionando.

| Mensaje | Qué hacer |
|---|---|
| "no está configurada: falta …" | Revise que exista `config\base_datos.ini` con `base_datos` y `cliente` |
| "no se encontró el cliente de Firebird (fbclient.dll)" | Corrija la ruta `cliente` |
| "no se pudo cargar el cliente de Firebird" | Use un `fbclient.dll` de 64 bits de Firebird 3 o posterior |
| "no se encontró el archivo de la base de datos" | Corrija `base_datos`. Si la base está en otro equipo, llene `host` |
| "no se pudo conectar a la base de datos" | Revise usuario y clave. Si el servidor Firebird tiene la base abierta, ponga `host = localhost` |
| "no se pudo leer la base de datos" | Verifique que sea la base de ventas del taller y que el usuario pueda leerla |
| "falta el paquete firebird-driver" | Cierre el programa y vuelva a abrir `Iniciar_Dashboard.bat` |

## Opcional: variables de entorno

En lugar del archivo se pueden usar las variables `FIREBIRD_DB`, `FIREBIRD_HOST`, `FIREBIRD_USER`, `FIREBIRD_PASSWORD`, `FIREBIRD_CLIENT` y `FIREBIRD_MESES`. Si existen, tienen prioridad sobre el archivo.
