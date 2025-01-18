import pandas as pd
import os
import numpy as np
from datetime import datetime

OUTPUT = "output"
STRING_TYPE = "object"


def excel_to_csv(dataset_file_path:str, file_name:str) -> pd.DataFrame:
    path = os.path.join(OUTPUT,file_name+".csv")
    df = pd.read_excel(dataset_file_path)
    df.to_csv(path, index = False)
    return df

def delete_unnecesary_information(df:pd.DataFrame) -> pd.DataFrame:
    df.drop(["GpsProvider", "BookingID", "vehicle_no", "Origin_Location", "Destination_Location",
             "Current_Location", "DestinationLocation", "OriginLocation_Code", "Driver_MobileNo",
             "DestinationLocation_Code", "customerNameCode", "supplierNameCode"], inplace=True, axis = 1)
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

def change_categoricall_to_numerical(df:pd.DataFrame) -> pd.DataFrame:
    pass


if __name__ == '__main__':
    df = excel_to_csv("Data/Delivery truck trip data.xlsx", "Dataset")
    df = delete_unnecesary_information(df)
    df = change_to_datetime(df)
    df = change_longitude_altitude(df)
    print(df.head())

