from pathlib import Path
import pandas as pd
import re


class DatasetCleaner:
    """
    Clase reutilizable para limpieza de datasets.
    Permite:
    - Leer archivos CSV
    - Filtrar registros por palabras clave
    - Eliminar datos no deseados
    - Guardar datasets limpios
    """

    def __init__(
        self,
        file_path,
        separator=';',
        encoding='utf-8'
    ):
        self.file_path = Path(file_path)
        self.separator = separator
        self.encoding = encoding
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(
            self.file_path,
            sep=self.separator,
            encoding=self.encoding
        )

        print("Dataset cargado correctamente.")
        print(f"Filas: {len(self.df)}")
        print(f"Columnas: {len(self.df.columns)}")

    def show_columns(self):
        print("\nColumnas del dataset:")
        for col in self.df.columns:
            print(f"- {col}")

    def remove_by_keywords(
        self,
        column_name,
        keywords,
        case_sensitive=False
    ):
        if self.df is None:
            raise Exception("Primero debes cargar el dataset.")

        original_rows = len(self.df)

        pattern = '|'.join(
            [re.escape(word) for word in keywords]
        )

        self.df = self.df[
            ~self.df[column_name].str.contains(
                pattern,
                case=case_sensitive,
                na=False
            )
        ].copy()

        removed_rows = original_rows - len(self.df)

        print("\n===== LIMPIEZA REALIZADA =====")
        print(f"Palabras eliminadas: {keywords}")
        print(f"Filas eliminadas: {removed_rows}")
        print(f"Filas restantes: {len(self.df)}")

    def remove_price_anomalies(
        self,
        price_column,
        qty_column,
        total_column,
        min_price=10,
    ):
        """
        Elimina registros con precio anómalo.

        Un registro es anómalo cuando:
          1. PRECIO_PROMEDIO <= min_price  (precio simbólico, ej: $1)
          2. VENTAS == TOTAL_VENDIDO       (unidades = total → sin precio real)

        Estos registros corresponden a cortesías, errores de digitación
        o productos sin precio configurado en el sistema (VALORU = 1).

        Parámetros
        ----------
        price_column : nombre de la columna de precio promedio
        qty_column   : nombre de la columna de cantidad vendida (VENTAS)
        total_column : nombre de la columna de total vendido (TOTAL_VENDIDO)
        min_price    : umbral mínimo de precio válido (default 10)
        """
        if self.df is None:
            raise Exception("Primero debes cargar el dataset.")

        original_rows = len(self.df)

        def _to_float(col):
            return (
                col.astype(str)
                .str.replace('.', '', regex=False)
                .str.replace(',', '.', regex=False)
                .astype(float)
            )

        precio = _to_float(self.df[price_column])
        ventas = _to_float(self.df[qty_column])
        total  = _to_float(self.df[total_column])

        mascara_anomalos = (precio <= min_price) | (ventas == total)
        self.df = self.df[~mascara_anomalos].copy()

        removed_rows = original_rows - len(self.df)

        print("\n===== LIMPIEZA DE PRECIOS ANÓMALOS =====")
        print(f"Umbral mínimo de precio: ${min_price}")
        print(f"Filas eliminadas: {removed_rows} ({removed_rows/original_rows*100:.1f}%)")
        print(f"Filas restantes: {len(self.df)}")

    def remove_bulk_adjustments(
        self,
        date_column,
        id_column,
        qty_column,
        price_column,
        min_products=100,
    ):
        """
        Envoltorio de conveniencia sobre remove_bulk_adjustments() (función
        de módulo, ver más abajo) que opera in-place sobre self.df y
        devuelve el resumen de filas eliminadas. Útil para reutilizar esta
        regla en scripts de limpieza offline como data/raw/cleaner.py.
        """
        if self.df is None:
            raise Exception("Primero debes cargar el dataset.")

        self.df, resumen = remove_bulk_adjustments(
            self.df, date_column, id_column, qty_column, price_column,
            min_products=min_products,
        )

        print("\n===== AJUSTES MASIVOS DE INVENTARIO ELIMINADOS =====")
        print(f"Umbral mínimo de productos distintos: {min_products}")
        print(f"Filas eliminadas: {resumen['rows_removed']} "
              f"({resumen['distinct_products']} productos distintos, "
              f"{len(resumen['affected_dates'])} fecha(s) afectada(s))")
        print(f"Monto excluido: ${resumen['cop_amount']:,.0f} COP")
        print(f"Filas restantes: {len(self.df)}")

        return resumen

    def save_dataset(
        self,
        output_path,
        encoding='utf-8-sig'
    ):
        self.df.to_csv(
            output_path,
            sep=self.separator,
            index=False,
            encoding=encoding
        )

        print("\nDataset guardado en:")
        print(output_path)


# ── Ajustes masivos de inventario (función de módulo, sin estado) ───────────
def remove_bulk_adjustments(
    df,
    date_col,
    id_col,
    qty_col,
    price_col,
    min_products=100,
):
    """
    Elimina filas de "ajuste masivo de inventario": aquellas cuya
    combinación EXACTA (fecha, cantidad, precio) es compartida por al menos
    min_products productos DISTINTOS ese mismo día.

    Motivo (HALLAZGO 2026-09-23, CONFIRMADO por el dueño del negocio como
    ajuste de inventario, no venta real): el 19/02/2026, 1.761 productos
    distintos (sin relación entre sí: tornillos, retenes, rodillos, etc.)
    registran EXACTAMENTE 12 unidades a 111 COP (1.332 COP) cada uno, una
    sola fila por producto. Es estadísticamente imposible que tantos
    productos distintos se vendan el mismo día con idéntica cantidad e
    idéntico precio; el patrón es propio de una carga o ajuste masivo del
    sistema de inventario, no de ventas reales a clientes. Representa apenas
    el 0,018 % de las ventas en pesos del dataset, pero ~5 % de los
    producto-mes con ventas: sin excluirlo, infla artificialmente la
    frecuencia de venta de 1.761 productos y crea un pico irreal en la
    distribución de ventas (ver histograma antes/después en el CHANGELOG).

    La regla es GENÉRICA (no hardcodea la fecha 19/02/2026, ni ninguna
    cantidad o precio) porque en el futuro el pipeline leerá directamente la
    base de datos de la empresa en vez de un CSV curado a mano, y un ajuste
    de este tipo puede repetirse con otra fecha, cantidad o precio.

    Por qué el umbral por defecto (min_products=100) es conservador: sobre
    data/raw/Query_Result_V5.csv, el siguiente grupo (fecha, cantidad,
    precio) más grande después del caso confirmado tiene solo 46 productos
    distintos — menos de la mitad del umbral. Top 5 grupos por número de
    productos distintos:
        1. 2026-02-19, 12 uds x 111 COP    -> 1.761 productos (caso confirmado)
        2. 2024-03-27, 20 uds x 1.111 COP  ->    46 productos
        3. 2024-03-27, 10 uds x 1.111 COP  ->    29 productos
        4. 2024-03-27, 40 uds x 1.111 COP  ->    22 productos
        5. 2024-04-12,  1 ud  x   100 COP  ->    19 productos
    Con min_products=100 la regla elimina EXACTAMENTE el caso confirmado y
    no toca ningún otro grupo de la V5 actual (el segundo grupo más grande,
    46, queda muy por debajo del umbral).

    Parámetros
    ----------
    df           : DataFrame de origen, a nivel de FILA (antes de agregar a
                   producto-mes). No se modifica: se devuelve una copia.
    date_col     : columna de fecha. Puede ser texto o datetime; solo se
                   usa para comparar igualdad dentro de cada grupo, no se
                   parsea.
    id_col       : columna de identificador de producto.
    qty_col      : columna de cantidad vendida.
    price_col    : columna de precio (promedio o unitario).
    min_products : mínimo de productos DISTINTOS que deben compartir la
                   combinación (fecha, cantidad, precio) para tratarla como
                   ajuste masivo (default 100). Se cuentan productos
                   distintos, no filas: varias filas del MISMO producto con
                   igual fecha/cantidad/precio no inflan el conteo.

    Devuelve
    --------
    (df_limpio, resumen) donde df_limpio es una copia de df sin las filas
    de ajuste masivo, y resumen es un dict con:
      "rows_removed"      : filas eliminadas.
      "distinct_products" : productos distintos afectados.
      "affected_dates"    : lista ordenada de valores únicos de date_col
                             entre las filas eliminadas.
      "cop_amount"        : suma de qty_col * price_col de las filas
                             eliminadas (monto en COP excluido).
    Si ningún grupo alcanza min_products, df_limpio es una copia idéntica
    de df y resumen queda en cero / listas vacías.
    """
    if df.empty:
        return df.copy(), {
            "rows_removed": 0,
            "distinct_products": 0,
            "affected_dates": [],
            "cop_amount": 0.0,
        }

    conteo_productos = df.groupby(
        [date_col, qty_col, price_col]
    )[id_col].transform("nunique")

    mascara_ajuste = conteo_productos >= min_products

    filas_eliminadas = df[mascara_ajuste]
    df_limpio = df[~mascara_ajuste].copy()

    resumen = {
        "rows_removed": int(mascara_ajuste.sum()),
        "distinct_products": int(filas_eliminadas[id_col].nunique()),
        "affected_dates": sorted(filas_eliminadas[date_col].unique().tolist()),
        "cop_amount": float(
            (filas_eliminadas[qty_col] * filas_eliminadas[price_col]).sum()
        ),
    }

    return df_limpio, resumen
