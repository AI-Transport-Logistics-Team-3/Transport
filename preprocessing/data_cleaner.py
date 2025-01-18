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
    df = df.drop(["GpsProvider", "BookingID", "vehicle_no", "Origin_Location", "Destination_Location",
             "Current_Location", "DestinationLocation", "OriginLocation_Code", "Driver_MobileNo",
             "DestinationLocation_Code", "customerNameCode", "supplierNameCode"], axis = 1)
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

def categorical_to_discret_values(df:pd.DataFrame) -> pd.DataFrame:
    for (c, t) in df.dtypes.items():
        print(c, t)
        if t == STRING_TYPE:
            print(c)
            print( c, df[c].dropna().apply(lambda x: "," in x).all() )
            #exit()



if __name__ == '__main__':
    df = excel_to_csv("Data/Delivery truck trip data.xlsx", "Dataset")
    df = delete_unnecesary_information(df)
    df = change_to_datetime(df)
    categorical_to_discret_values(df)

