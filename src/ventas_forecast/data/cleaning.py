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
