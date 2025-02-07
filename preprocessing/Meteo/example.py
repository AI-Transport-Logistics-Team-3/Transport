import os
# import time
# import sys

# import numpy as np
import pandas as pd

# from datetime import datetime
# from typing import Iterator
from sklearn.preprocessing import LabelEncoder
from meteo_forecast import MeteoData
from math import radians, sin, cos, sqrt, atan2

DATASET = "Delivery truck trip data.xlsx"
INPUT =  "Data"
OUTPUT = "output"
dataset_file_path = os.path.join(INPUT, DATASET)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  #radio de la tierra

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

	# Fórmula de Haversine
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c

    return distance



ruta_file = "Data\\Delivery truck trip data.xlsx"
df = pd.read_excel(ruta_file)  # Leer archivo Excel
df.columns = df.columns.str.rstrip()  # Eliminar espacios en los nombres de las columnas
    

df = df.drop(df[
    (df['Planned_ETA'].isna())   
    | (df['actual_eta'].isna())   
    | (pd.to_datetime(df['Planned_ETA']).dt.year < 2019) 
    | (pd.to_datetime(df['actual_eta']).dt.year < 2019)
    | (df['actual_eta']< df['trip_start_date']) 
    | (df['ontime'].notnull() & df['delay'].notnull())
   ].index)

df['ontime_delay'] = df['ontime'].fillna(df['delay'])
df['ontime_delay'] = df['ontime_delay'].replace('R', 0)
df['ontime_delay'] = df['ontime_delay'].replace('G', 1)

df[['Org_latitude', 'Org_longitude']] = df['Org_lat_lon'].str.split(',', expand=True)
df[['Des_latitude', 'Des_longitude']] = df['Des_lat_lon'].str.split(',', expand=True)
df['Org_latitude'] = pd.to_numeric(df['Org_latitude'])  
df['Org_longitude'] = pd.to_numeric(df['Org_longitude'])  
df['Des_latitude'] = pd.to_numeric(df['Des_latitude'])  
df['Des_longitude'] = pd.to_numeric(df['Des_longitude'])  

df['MT'] = df['vehicleType'].str.extract(r'(\d+)(?=\s*MT)')
df['MT'] = pd.to_numeric(df['MT'].fillna('0'))

label_encoder = LabelEncoder()
df['vehicleType'] = label_encoder.fit_transform(df['vehicleType'].fillna('Unknown'))
df['customerID'] = label_encoder.fit_transform(df['customerID'])
df['supplierID'] = label_encoder.fit_transform(df['supplierID'].astype(str))
df['Material Shipped'] = label_encoder.fit_transform(df['Material Shipped'].fillna('Unknown'))


df['ID'] = df.index
df['trip_start_date_formatted'] =  pd.to_datetime(df['trip_start_date']).dt.strftime('%Y-%m-%d')
df['actual_eta_formatted'] = pd.to_datetime(df['actual_eta']).dt.strftime('%Y-%m-%d') 
df_clima = df[['ID','Org_latitude', 'Org_longitude','Des_latitude', 'Des_longitude','trip_start_date_formatted','actual_eta_formatted']]
df_clima.columns = ['ID', 'latitude', 'longitude', 'latitudedest', 'longitudedest', 'startdate', 'enddate']

meteo = MeteoData(df_clima)
result= meteo.fetch_weather_data()
df = df.merge(result[["ID","weather_code", "temperature_max","temperature_min"]], on='ID', how='left')

# print(df[["ID","weather_code", "temperature_max","temperature_min"]])
# print(meteo.get_api_calls())
# print(meteo.get_from_internal_db())
# print(meteo.get_errors().size)
meteo.get_errors().to_csv("C:\Sonia\ProyectoFinal\Transport\output\errores.csv")


df["diff_hours"] = (pd.to_datetime(df['actual_eta'])- pd.to_datetime(df['Planned_ETA']))/ pd.Timedelta(hours=1)
df['Planned_ETA'] = pd.to_datetime(df['Planned_ETA'])
df['actual_eta'] = pd.to_datetime(df['actual_eta'])
df['planned_day_of_week'] = df['Planned_ETA'].dt.dayofweek
df['actual_day_of_week'] = df['actual_eta'].dt.dayofweek
df['planned_hour'] = df['Planned_ETA'].dt.hour
df['actual_hour'] = df['actual_eta'].dt.hour
df['planned_month'] = df['Planned_ETA'].dt.month
df['actual_month'] = df['actual_eta'].dt.month

df['Origin_Location'] = df['Origin_Location'].str.lower()
df['origin_state'] = df['Origin_Location'].apply((lambda x: x.split(',')[-1]))
df['origin_state'] = df['origin_state'].str.lstrip()

non_india = df[(df['Origin_Location'].apply((lambda x: x.split(',')[-1])) != ' india').values]['Origin_Location'].apply(lambda x: x.split(',')[-1])
with_india = df[(df['Origin_Location'].apply((lambda x: x.split(',')[-1])) == ' india').values]['Origin_Location'].apply(lambda x: x.split(',')[-2])
df['origin_state'] = pd.concat([non_india, with_india], axis=0)
df['origin_state'] = df['origin_state'].str.lstrip()
df['origin_state'] = df['origin_state'].replace('pondicherry', 'puducherry')
df['origin_state'] = df['origin_state'].replace('karnataka 562114', 'karnataka')
df['origin_state'] = df['origin_state'].replace('chattisgarh', 'Chhattisgarh')
df['origin_state'] = df['origin_state'].replace('sedarapet', 'puducherry')

df['Destination_Location'] = df['Destination_Location'].str.lower()
df['dest_state'] = df['Destination_Location'].apply((lambda x: x.split(',')[-1]))
df['dest_state'] = df['dest_state'].str.lstrip()
non_india = df[(df['Destination_Location'].apply((lambda x: x.split(',')[-1])) != ' india').values]['Destination_Location'].apply(lambda x: x.split(',')[-1])
with_india = df[(df['Destination_Location'].apply((lambda x: x.split(',')[-1])) == ' india').values]['Destination_Location'].apply(lambda x: x.split(',')[-2])
df['dest_state'] = pd.concat([non_india, with_india], axis=0)
df['dest_state'] = df['dest_state'].str.lstrip()
df['dest_state'] = df['dest_state'].replace('pondicherry', 'puducherry')
df['dest_state'] = df['dest_state'].replace('karnataka 560300', 'karnataka')
df['dest_state'] = df['dest_state'].replace('chattisgarh', 'chhattisgarh')
df['dest_state'] = df['dest_state'].replace('jammu & kashmir', 'Jammu and Kashmir')


df['origin_state'] = label_encoder.fit_transform(df['origin_state'].fillna('Unknown'))
df['dest_state'] = label_encoder.fit_transform(df['dest_state'].fillna('Unknown'))

df['distance'] = df.apply(lambda row: row['TRANSPORTATION_DISTANCE_IN_KM'] if pd.notna(row['TRANSPORTATION_DISTANCE_IN_KM']) else haversine(row['Org_latitude'], row['Org_latitude'], row['Des_latitude'], row['Des_longitude']), axis=1)

df.drop(["GpsProvider", "BookingID", "vehicle_no", "Origin_Location", "Destination_Location",
             "Current_Location", "DestinationLocation", "OriginLocation_Code", "Driver_MobileNo", "Market/Regular",
             "DestinationLocation_Code", "customerNameCode", "supplierNameCode", "Driver_Name" ,"BookingID_Date",
             "Org_lat_lon","Des_lat_lon","Data_Ping_time", "Planned_ETA", "actual_eta", "Curr_lat","Curr_lon"
             , "ontime","delay" ,"trip_start_date","trip_end_date","Minimum_kms_to_be_covered_in_a_day","trip_start_date_formatted",
             "actual_eta_formatted", "TRANSPORTATION_DISTANCE_IN_KM"], inplace=True, axis=1)

path = "C:\\Sonia\\ProyectoFinal\\Transport\\output\\clean_dataset.csv"
df.to_csv(path, index=False)