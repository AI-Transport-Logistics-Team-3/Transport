# data_preprocessing.py
#
# Autor: Gerard Lahuerta Martín
# GitHub: @Gerard-Lahuerta
# Fecha de creación: 18-01-2025
# Descripción: Este script realiza un preprocesamiento de datos de un archivo Excel en concreto: "Delivery truck trip data.xlsx".
#
# Dependencias:
# - pandas
# - numpy
# - openpyxl
#
# Nota: Este código puede ser importado y utilizado en otros archivos Python.

import os
import time
import threading
import sys

import numpy as np
import pandas as pd

from datetime import datetime
from typing import Iterator

# Constantes de entrada, salida, y formato de datos
INPUT = "Data"
OUTPUT = "output"
STRING_TYPE = "object"
DATASET = "Delivery truck trip data.xlsx"

# Función para convertir un archivo Excel a CSV y opcionalmente guardarlo
def _excel_to_csv(dataset_file_path: str, file_name: str = "Dataset", save: bool = False) -> pd.DataFrame:
    """
    Convierte un archivo Excel a un DataFrame de Pandas y lo guarda como un archivo CSV si se especifica.

    Parameters:
    - dataset_file_path (str): Ruta del archivo Excel de entrada.
    - file_name (str): Nombre del archivo CSV resultante. Por defecto es "Dataset".
    - save (bool): Si es True, guarda el DataFrame como un archivo CSV.

    Returns:
    - pd.DataFrame: El DataFrame cargado desde el archivo Excel.
    """
    df = pd.read_excel(dataset_file_path)  # Leer archivo Excel
    df.columns = df.columns.str.rstrip()  # Eliminar espacios en los nombres de las columnas
    if save:  # Si 'save' es True, guarda el DataFrame como archivo CSV
        path = os.path.join(OUTPUT, file_name + ".csv")
        df.to_csv(path, index=False)
    return df

# Función para eliminar columnas innecesarias
def delete_unnecesary_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina columnas innecesarias del DataFrame.

    Parameters:
    - df (pd.DataFrame): El DataFrame original.

    Returns:
    - pd.DataFrame: El DataFrame con las columnas innecesarias eliminadas.
    """
    # Eliminar columnas específicas del DataFrame
    df.drop(["GpsProvider", "BookingID", "vehicle_no", "Origin_Location", "Destination_Location",
             "Current_Location", "DestinationLocation", "OriginLocation_Code", "Driver_MobileNo", "Market/Regular",
             "DestinationLocation_Code", "customerNameCode", "supplierNameCode", "Driver_Name"], inplace=True, axis=1)
    return df

# Generador para convertir fechas a timestamp (unix)
def _to_timestamp(dates: list[datetime]) -> Iterator[float]:
    """
    Convierte fechas a formato timestamp (unix).

    Parameters:
    - dates (list[datetime]): Lista de objetos datetime.

    Yields:
    - float: El timestamp de cada fecha.
    """
    for d in dates:
        try:
            yield d.timestamp()
        except:
            yield 0.0

# Función para convertir columnas de fechas a formato timestamp
def change_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte las columnas de fechas en el DataFrame a formato timestamp (unix).

    Parameters:
    - df (pd.DataFrame): El DataFrame con datos.

    Returns:
    - pd.DataFrame: El DataFrame con las columnas de fechas convertidas a timestamp.
    """
    df_len = df.shape[0]  # Longitud del DataFrame
    for (c, t) in df.dtypes.items():  # Itera por las columnas
        reg = df[c].dropna().apply(lambda x: isinstance(x, datetime)).sum()
        if reg > df_len // 2:  # Si más de la mitad de los valores son fechas, convierte a timestamp
            df[c] = np.array([i for i in _to_timestamp(df[c].to_list())])
    return df

# Función para separar coordenadas de latitud y longitud en columnas separadas
def change_longitude_altitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Separa las coordenadas de latitud y longitud contenidas en una columna en dos columnas separadas.

    Parameters:
    - df (pd.DataFrame): El DataFrame con las coordenadas.

    Returns:
    - pd.DataFrame: El DataFrame con las coordenadas separadas en dos columnas.
    """
    for (c, t) in df.dtypes.items():
        if t == STRING_TYPE:
            if df[c].dropna().str.contains(',').all():  # Si la columna contiene coordenadas separadas por coma
                col_name = c.split("_")[0]  # Obtener nombre de la columna antes del '_'
                df[[f'{col_name}_latitude', f'{col_name}_longitude']] = df[c].str.split(',', expand=True)
                df[f'{col_name}_latitude'] = pd.to_numeric(df[f'{col_name}_latitude'])  # Convertir a numérico
                df[f'{col_name}_longitude'] = pd.to_numeric(df[f'{col_name}_longitude'])  # Convertir a numérico
                df.drop(c, inplace=True, axis=1)  # Eliminar la columna original
    return df

# Función de codificación binaria (0 o 1) para columnas con valores no nulos o nulos
def _binary_encoding(reg: pd.Series) -> pd.Series:
    """
    Convierte una serie en una codificación binaria: 0 para valores nulos, 1 para valores no nulos.

    Parameters:
    - reg (pd.Series): La serie de valores a convertir.

    Returns:
    - pd.Series: La serie codificada.
    """
    return reg.apply(lambda x: 0 if pd.isna(x) else 1)

# Función de codificación de etiquetas para columnas con 2 valores únicos
def _label_encoding(reg: pd.Series) -> pd.Series:
    """
    Convierte una serie con dos valores únicos en una codificación de etiquetas numéricas.

    Parameters:
    - reg (pd.Series): La serie con dos valores únicos.

    Returns:
    - pd.Series: La serie codificada con números.
    """
    return reg.astype('category').cat.codes.nunique() - 1

# Función de codificación one-hot para columnas con más de 2 valores únicos
def _one_hotter_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Convierte una columna categórica con más de 2 valores únicos en codificación one-hot.

    Parameters:
    - df (pd.DataFrame): El DataFrame con la columna.
    - col (str): El nombre de la columna a codificar.

    Returns:
    - pd.DataFrame: El DataFrame con la columna codificada en one-hot.
    """
    return pd.get_dummies(df, columns=[col])

# Función para limpiar las columnas de tipo "OTROS" y agrupar los valores pequeños
def _clean_dataset(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Agrupa las columnas con el sufijo indicado en la columna "OTROS" si tienen pocos valores.

    Parameters:
    - df (pd.DataFrame): El DataFrame con las columnas a limpiar.
    - col (str): El prefijo de las columnas a limpiar.

    Returns:
    - pd.DataFrame: El DataFrame con las columnas agrupadas en "OTROS".
    """
    otros = col + "_OTROS"  # Nombre de la columna para los valores agrupados
    df_len = df.shape[0]  # Longitud del DataFrame
    threshold = df_len * 0.05  # Umbral para agrupar los valores
    columns = df.columns
    for column in columns:
        if col in column:  # Si el nombre de la columna contiene el prefijo dado
            if df[column].sum() < threshold:  # Si la suma de los valores es menor que el umbral
                if otros not in df.columns:  # Si no existe una columna "OTROS", se crea
                    df[otros] = np.array([False] * df_len)
                df[otros] += df[column]  # Agrupar el valor en "OTROS"
                df.drop(column, axis=1, inplace=True)  # Eliminar la columna original
    return df

# Función para convertir columnas categóricas a numéricas
def change_categoricall_to_numerical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte todas las columnas categóricas a formato numérico mediante diferentes técnicas de codificación.

    Parameters:
    - df (pd.DataFrame): El DataFrame con las columnas a convertir.

    Returns:
    - pd.DataFrame: El DataFrame con las columnas categóricas convertidas a valores numéricos.
    """
    items = df.dtypes.items()
    for (c, t) in items:
        if t == STRING_TYPE:
            if df[c].nunique() == 1:  # Si la columna tiene un solo valor único
                df[c] = _binary_encoding(df[c])  # Aplicar codificación binaria
            elif df[c].nunique() == 2:  # Si la columna tiene dos valores únicos
                df[c] = _label_encoding(df[c])  # Aplicar codificación de etiquetas
            else:
                df = _one_hotter_encoding(df, c)  # Aplicar codificación one-hot
                df = _clean_dataset(df, c)  # Limpiar las columnas "OTROS"
    df.columns = df.columns.str.rstrip()  # Eliminar espacios al final de los nombres de columnas
    df.columns = df.columns.str.replace(' ', '_')  # Reemplazar los espacios por guiones bajos
    return df

# Función para normalizar las columnas numéricas
def normalice_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza todas las columnas numéricas del DataFrame utilizando la normalización Z-score.

    Parameters:
    - df (pd.DataFrame): El DataFrame con las columnas a normalizar.

    Returns:
    - pd.DataFrame: El DataFrame con las columnas numéricas normalizadas.
    """
    for c in df.columns:
        if df[c].nunique() > 2:  # Normalizar solo columnas con más de dos valores únicos
            df[c] = (df[c] - df[c].mean()) / df[c].std()  # Normalización Z-score
    return df

# Función para mostrar un spinner mientras se ejecuta el procesamiento
def _spinner(stop_event) -> None:
    """
    Muestra un spinner (animación) en la consola mientras se realiza un proceso.

    Parameters:
    - stop_event (threading.Event): Evento que se utiliza para detener el spinner.

    Este es un hilo de fondo que ejecuta un ciclo de caracteres de spinner y lo muestra en la consola.
    """
    spinner_chars = ['|', '/', '-', '\\']
    while not stop_event.is_set():  # El spinner sigue mientras no se detenga
        for char in spinner_chars:
            if stop_event.is_set():
                break
            sys.stdout.write(f'\r{char} Ejecutando...')  # Mostrar el carácter del spinner
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write('\Procesado de datos completado.          \n')  # Mensaje cuando termina el proceso

# Clase principal para el procesamiento de datos
class data_preprocessing():
    """
    Clase para manejar y realizar el preprocesamiento de datos a partir de un archivo Excel, 
    incluyendo la eliminación de columnas innecesarias, conversión de tipos y normalización.

    Attributes:
    - original (pd.DataFrame): El DataFrame original cargado desde el archivo.
    - dataset (pd.DataFrame): El DataFrame procesado después de eliminar las columnas innecesarias y convertir tipos.
    - normaliced (pd.DataFrame): El DataFrame con los datos normalizados.

    Methods:
    - get_processed_dataset: Retorna el DataFrame procesado.
    - get_normaliced_dataset: Retorna el DataFrame normalizado.
    - get_orgiginal_dataset: Retorna el DataFrame original.
    - get_original_data: Retorna las filas originales para un índice dado.
    - save_preprocessed_dataset: Guarda el DataFrame procesado en un archivo.
    - save_normaliced_dataset: Guarda el DataFrame normalizado en un archivo.
    - save_original_dataset: Guarda el DataFrame original en un archivo.
    """

    def __init__(self, input: str = INPUT, dataset: str = DATASET, file_name: str = "Dataset") -> None:
        """
        Inicializa la clase y comienza el procesamiento de datos.

        Parameters:
        - input (str): Carpeta de entrada donde se encuentra el archivo de datos.
        - dataset (str): Nombre del archivo de datos a procesar.
        - file_name (str): Nombre del archivo para guardar el resultado (opcional).

        Este constructor inicia un hilo para el spinner de carga y realiza el procesamiento 
        de los datos a través de varias etapas (carga, limpieza, conversión y normalización).
        """
        stop_event = threading.Event()  # Crear un evento para detener el spinner
        
        # Crear un hilo para el spinner
        spinner_thread = threading.Thread(target=_spinner, args=(stop_event,))
        spinner_thread.start()
        
        # Procesamiento de datos (simulado aquí con una secuencia de operaciones)
        df = _excel_to_csv(os.path.join(input, dataset), file_name)  # Cargar el dataset
        self.original = df.copy()  # Guardar el dataset original
        df = delete_unnecesary_information(df)  # Eliminar columnas innecesarias
        df = change_to_datetime(df)  # Convertir las fechas a timestamp
        df = change_longitude_altitude(df)  # Separar coordenadas de latitud y longitud
        df = change_categoricall_to_numerical(df)  # Convertir columnas categóricas a numéricas
        self.dataset = df.copy()  # Guardar el dataset procesado
        df = df.replace({True: 1, False: 0})  # Reemplazar valores booleanos por enteros
        df = normalice_dataset(df)  # Normalizar las columnas numéricas
        self.normaliced = df.copy()  # Guardar el dataset normalizado
        
        # Detener el spinner cuando el proceso haya terminado
        stop_event.set()
        
        # Esperar que el hilo del spinner termine
        spinner_thread.join()

    # Métodos para obtener los datasets procesados
    def get_processed_dataset(self) -> pd.DataFrame:
        """
        Retorna el DataFrame procesado (sin valores originales).

        Returns:
        - pd.DataFrame: El DataFrame después de los pasos de limpieza y conversión.
        """
        return self.dataset
    
    def get_normaliced_dataset(self) -> pd.DataFrame:
        """
        Retorna el DataFrame normalizado.

        Returns:
        - pd.DataFrame: El DataFrame después de la normalización.
        """
        return self.normaliced
    
    def get_orgiginal_dataset(self) -> pd.DataFrame:
        """
        Retorna el DataFrame original sin modificaciones.

        Returns:
        - pd.DataFrame: El DataFrame original cargado desde el archivo.
        """
        return self.original
    
    # Función para obtener los datos originales usando los indices de los datos procesados
    def get_original_data(self, index: pd.Index) -> pd.DataFrame:
        """
        Retorna las filas originales para un índice dado.

        Parameters:
        - index (pd.Index): Índices de las filas originales que se quieren obtener.

        Returns:
        - pd.DataFrame: Las filas originales correspondientes a los índices proporcionados.
        """
        return self.original.iloc[index]
    
    # Función interna para verificar el formato de archivo
    def _check_format(name: str, format: str, dir: str):
        """
        Verifica si el formato del archivo es válido y genera la ruta completa del archivo.

        Parameters:
        - name (str): Nombre del archivo (sin extensión).
        - format (str): Formato del archivo (debe ser 'csv' o 'xlsx').
        - dir (str): Directorio donde guardar el archivo.

        Returns:
        - str: La ruta completa del archivo.
        
        Raises:
        - ValueError: Si el formato del archivo no es válido.
        """
        if format not in ['csv', 'xlsx']:  # Verificar si el formato es válido
            raise ValueError(f"Formato del fichero inválido: {format}. Debe ser 'csv' o 'xlsx'.")
        return os.path.join(dir, name + "." + format)
    
    # Métodos para guardar los datasets en diferentes formatos
    def save_preprocessed_dataset(self, name: str = "original", format: str = "csv", dir: str = OUTPUT):
        """
        Guarda el DataFrame procesado en un archivo con el nombre y formato especificados.

        Parameters:
        - name (str): Nombre del archivo.
        - format (str): Formato del archivo ('csv' o 'xlsx').
        - dir (str): Directorio donde guardar el archivo.
        """
        path = self._check_format(name, format, dir)
        if format == "csv":
            self.dataset.to_csv(path, index=False)
        if format == "xlsx":
            self.dataset.to_excel(path, index=False)
    
    def save_normaliced_dataset(self, name: str = "original", format: str = "csv", dir: str = OUTPUT):
        """
        Guarda el DataFrame normalizado en un archivo con el nombre y formato especificados.

        Parameters:
        - name (str): Nombre del archivo.
        - format (str): Formato del archivo ('csv' o 'xlsx').
        - dir (str): Directorio donde guardar el archivo.
        """
        path = self._check_format(name, format, dir)
        if format == "csv":
            self.normaliced.to_csv(path, index=False)
        if format == "xlsx":
            self.normaliced.to_excel(path, index=False)

    def save_original_dataset(self, name: str = "original", format: str = "csv", dir: str = OUTPUT):
        """
        Guarda el DataFrame original en un archivo con el nombre y formato especificados.

        Parameters:
        - name (str): Nombre del archivo.
        - format (str): Formato del archivo ('csv' o 'xlsx').
        - dir (str): Directorio donde guardar el archivo.
        """
        path = self._check_format(name, format, dir)
        if format == "csv":
            self.original.to_csv(path, index=False)
        if format == "xlsx":
            self.original.to_excel(path, index=False)


# Bloque principal para ejecutar el preprocesamiento  y ejemplo de uso de la classe data_preprocessing
if __name__ == '__main__':
    dp = data_preprocessing()  # Inicializar el procesamiento de datos
    df = dp.get_normaliced_dataset()  # Obtener el dataset normalizado
    subset = df[df["supplierID_999"] == True]  # Filtrar por una condición
    original_data = dp.get_original_data(subset.index)  # Obtener los datos originales correspondientes
