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
