import sys
import threading
import time
from typing import Tuple
import pandas as pd
import os
import numpy as np
from datetime import datetime

INPUT = "Data"
OUTPUT = "output"
STRING_TYPE = "object"
DATASET = "Delivery truck trip data.xlsx"


def excel_to_csv(dataset_file_path:str, file_name:str = "Dataset", save:bool = False) -> pd.DataFrame:
    path = os.path.join(OUTPUT,file_name+".csv")
    df = pd.read_excel(dataset_file_path)
    df.columns = df.columns.str.rstrip()
    if save: df.to_csv(path, index = False)
    return df

def delete_unnecesary_information(df:pd.DataFrame) -> pd.DataFrame:
    df.drop(["GpsProvider", "BookingID", "vehicle_no", "Origin_Location", "Destination_Location",
             "Current_Location", "DestinationLocation", "OriginLocation_Code", "Driver_MobileNo", "Market/Regular",
             "DestinationLocation_Code", "customerNameCode", "supplierNameCode", "Driver_Name"], inplace=True, axis = 1)
    return df

def _to_timestamp(dates:list[datetime]):
    for d in dates:
        try:
            yield d.timestamp()
        except:
            yield 0

def change_to_datetime(df:pd.DataFrame) -> pd.DataFrame:
    df_len = df.shape[0]
    for (c, t) in df.dtypes.items():
        reg = df[c].dropna().apply(lambda x: isinstance(x, datetime)).sum()
        if reg > df_len//2:
            df[c] = np.array([ i for i in _to_timestamp(df[c].to_list())])
    return df

def change_longitude_altitude(df:pd.DataFrame) -> pd.DataFrame:
    for (c, t) in df.dtypes.items():
        if t == STRING_TYPE:
            if df[c].dropna().str.contains(',').all():
                col_name = c.split("_")[0]
                df[[f'{col_name}_latitude', f'{col_name}_longitude']] = df[c].str.split(',', expand=True)
                df[f'{col_name}_latitude'] = pd.to_numeric(df[f'{col_name}_latitude'])
                df[f'{col_name}_longitude'] = pd.to_numeric(df[f'{col_name}_longitude'])
                df.drop(c, inplace=True, axis = 1)
    return df

def _binary_encoding(reg: pd.Series) -> pd.Series:
    return reg.apply(lambda x: 0 if pd.isna(x) else 1)

def _label_encoding(reg: pd.Series) -> pd.Series:
    return reg.astype('category').cat.codes.nunique() - 1

def _one_hotter_encoding(df: pd.DataFrame, col:str) -> pd.DataFrame:
    return pd.get_dummies(df, columns=[col])

def _clean_dataset(df: pd.DataFrame, col:str) -> pd.DataFrame:
    otros = col+"_OTROS"
    df_len = df.shape[0]
    threshold = df_len * 0.05
    columns = df.columns
    for column in columns:
        if col in column:
            if df[column].sum() < threshold:
                if otros not in df.columns:
                    df[otros] = np.array([False]*df_len)
                df[otros] += df[column]
                df.drop(column, axis=1, inplace=True)
    return df

def change_categoricall_to_numerical(df:pd.DataFrame) -> pd.DataFrame:
    items = df.dtypes.items()
    for (c, t) in items:
        if t == STRING_TYPE:
            if df[c].nunique() == 1:
                df[c] = _binary_encoding(df[c])
            elif df[c].nunique() == 2:
                df[c] = _label_encoding(df[c])
            else:
                df = _one_hotter_encoding(df, c)
                df = _clean_dataset(df, c)
    df.columns = df.columns.str.rstrip()
    df.columns = df.columns.str.replace(' ', '_')
    return df


def normalice_dataset(df:pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].nunique() > 2:
            df[c] = (df[c] - df[c].mean()) / df[c].std()
    return df


def _spinner(stop_event):
    spinner_chars = ['|', '/', '-', '\\']
    while not stop_event.is_set():  # El spinner sigue mientras no se detenga
        for char in spinner_chars:
            if stop_event.is_set():
                break
            sys.stdout.write(f'\r{char} Ejecutando...')
            sys.stdout.flush()
            time.sleep(0.1)
    sys.stdout.write('\rProceso completado.          \n')



class data_preprocessing():
    def __init__(self, input:str = INPUT, dataset:str = DATASET, file_name:str = "Dataset") -> None:
        stop_event = threading.Event()
        
        # Crear un hilo para el spinner
        spinner_thread = threading.Thread(target=_spinner, args=(stop_event,))
        spinner_thread.start()
        
        # Procesamiento de datos (simulado aquí con una secuencia de operaciones)
        df = excel_to_csv(os.path.join(input, dataset), file_name)
        self.original = df.copy()
        df = delete_unnecesary_information(df)
        df = change_to_datetime(df)
        df = change_longitude_altitude(df)
        df = change_categoricall_to_numerical(df)
        self.dataset = df.copy()
        df = df.replace({True: 1, False: 0})
        df = normalice_dataset(df)
        self.normaliced = df.copy()
        
        # Detener el spinner cuando el proceso haya terminado
        stop_event.set()
        
        # Esperar que el hilo del spinner termine
        spinner_thread.join()

    def get_processed_dataset(self) -> pd.DataFrame:
        return self.dataset
    
    def get_normaliced_dataset(self) -> pd.DataFrame:
        return self.normaliced
    
    def get_orgiginal_dataset(self) -> pd.DataFrame:
        return self.original
    
    def get_original_data(self, index: pd.Index) -> pd.DataFrame:
        return self.original.iloc[index]
    
    def _check_format(name:str, format:str, dir:str):
        if format not in ['csv', 'xlsx']:
            raise ValueError(f"Formato del fichero inválido: {format}. Debe ser 'csv' o 'xlsx'.")
        return os.path.join(dir, name + "." + format)
    
    def save_preprocessed_dataset(self, name:str = "original", format:str = "csv", dir:str = OUTPUT):
        path = self._check_format(name, format, dir)
        if format == "csv": self.dataset.to_csv(path, index = False)
        if format == "xlsx": self.dataset.to_excel(path, index = False)
    
    def save_normaliced_dataset(self, name:str = "original", format:str = "csv", dir:str = OUTPUT):
        path = self._check_format(name, format, dir)
        if format == "csv": self.normaliced.to_csv(path, index = False)
        if format == "xlsx": self.normaliced.to_excel(path, index = False)

    def save_original_dataset(self, name:str = "original", format:str = "csv", dir:str = OUTPUT):
        path = self._check_format(name, format, dir)
        if format == "csv": self.original.to_csv(path, index = False)
        if format == "xlsx": self.original.to_excel(path, index = False)


    


if __name__ == '__main__':

    dp = data_preprocessing()
    df = dp.get_normaliced_dataset()
    subset = df[df["supplierID_999"] == True]
    original_data = dp.get_original_data(subset.index)

