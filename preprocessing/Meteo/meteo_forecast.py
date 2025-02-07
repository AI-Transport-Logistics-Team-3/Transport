# #pip install openmeteo-requests
# #pip install requests-cache retry-requests numpy pandas
# #https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM
# #https://open-meteo.com/

import openmeteo_requests
import requests_cache
import pandas as pd
import os

from retry_requests import retry
from datetime import datetime
from dateutil.relativedelta import relativedelta
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm

class MeteoDataProcessor:
    pass

class MeteoData(MeteoDataProcessor):
    API = "https://api.open-meteo.com/v1/forecast"
    API_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
    DATASET_NAME = "meteo_dataset.csv"
    DATASET_PATH = "Data"
    WMO_CODES = "wmo_code_table.csv"
    DIST_ROUTE_KM = 100
    TODAY = datetime.now()
    CURRENT_LIMIT_METEO = TODAY - relativedelta(months=3)
    
    def __init__(self, input_dataframe):
        """
        Initialize the MeteoData class with an input DataFrame.
        input_dataframe: DataFrame with columns ['ID', 'latitude', 'longitude',  'latitudedest', 'longitudedest',"startdate", "enddate"]
        if "enddate" is empty it will assume startdate=enddate
        if startdate > enddate it will take enddate as startdate
        """
        self.input_dataframe = input_dataframe

        # Setup the Open-Meteo API client with caching and retry logic
        self.cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        self.retry_session = retry(self.cache_session, retries=5, backoff_factor=0.2)
        self.client = openmeteo_requests.Client(session=self.retry_session)

        self.current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_path = os.path.join(self.DATASET_PATH, self.DATASET_NAME)

        if os.path.exists(self.dataset_path):
            self.dataset = pd.read_csv(self.dataset_path)
        else:
            self.dataset = pd.DataFrame(columns=["latitude", "longitude", "startdate", "enddate", "weather_code", "temperature_max","temperature_min"])

        wmo_path = os.path.join(self.DATASET_PATH, self.WMO_CODES)
        self.wmo_codes = pd.read_csv(wmo_path, sep=';')
        
        self.errors = pd.DataFrame(columns=["ID","Error"])
        self.api_calls = 0
        self.from_internal_db = 0

    def _save_dataset(self):
        """
        Save the dataset to a CSV file.
        """
        os.makedirs(os.path.join(self.DATASET_PATH), exist_ok=True)
        self.dataset.to_csv(self.dataset_path, index=False)
        
    def _haversine(self, lat1, lon1, lat2, lon2):
        # Radio de la Tierra en kilómetros
        R = 6371.0  

		# Convertir grados a radianes
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Diferencias entre las coordenadas
        dlat = lat2 - lat1
        dlon = lon2 - lon1

		# Fórmula de Haversine
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c

        return distance
    
    def get_weather_code_info(self, weather_code)-> str:
        """
        Return weather code desciption from wmo code table
        """
        description = self.wmo_codes.loc[self.wmo_codes["Code"] == weather_code, "Description"].values
        description = description[0] if len(description) > 0 else "Unknown"
        return description
    
    def get_weather_code_table(self):
        """
        Return weather codes table
        """
        return self.wmo_codes
    
    def get_errors(self):
        """
        Return weather codes table
        """
        return self.errors
    
    def get_api_calls(self):
        return self.api_calls
    
    def get_from_internal_db(self):
        return self.from_internal_db
    
    def _has_any_error(self, row) -> bool:
        """
        Checks if there are any error in the parameters
        """
        error=""
        has_error = False
        if pd.isnull(row["ID"]) or row["ID"] == "": error = "ID is mandatory"
        elif pd.isnull(row["startdate"]) or row["startdate"] == "" : error = "startdate is mandatory"
        elif pd.isnull(row["latitude"]) or row["latitude"] == "": error = "latitude is mandatory"
        elif pd.isnull(row["longitude"]) or row["longitude"] == "": error = "longitude is mandatory"
        if error != "":
            self.errors = pd.concat([
                                self.errors,
                                pd.DataFrame({
                                               "ID": [row["ID"]],
                                                "Error": [error]
                                            })
                                    ], ignore_index=True)
            has_error = True
        return has_error
    
    def _get_meteo_info(self, row, result_list):
                params = {
                    "latitude": row['latitude'],
                    "longitude": row['longitude'],
                    "daily": [
                        "weather_code", 
                        "temperature_2m_max", 
                        "temperature_2m_min" 
                    ],
                    "timezone": "auto",
                    "start_date": row['startdate'] ,
                    "end_date": row["enddate"] 
                }

                try:
                    # Request weather data from the API
                    
                    if pd.to_datetime(row["enddate"]) < pd.to_datetime(self.CURRENT_LIMIT_METEO):
                        responses = self.client.weather_api(self.API_ARCHIVE, params=params)
                    else:
                        responses = self.client.weather_api(self.API, params=params)
                    response = responses[0]  # Assuming one response per request
                    self.api_calls = self.api_calls +1
                    # Extract relevant data from the response
                    daily = response.Daily()
                    
                    weather_code = int(daily.Variables(0).ValuesAsNumpy().max())  
                    temp_max = int(daily.Variables(1).ValuesAsNumpy().max())
                    temp_min = int(daily.Variables(2).ValuesAsNumpy().min())                   
                    
                    if pd.to_datetime(row["enddate"]) < self.TODAY:  
                        self.dataset = pd.concat([
                                          self.dataset,
                                          pd.DataFrame({
                                          "latitude": [row["latitude"]],
                                          "longitude": [row["longitude"]],
                                          "startdate": [row["startdate"]] ,
                                          "enddate": [row["enddate"]] ,
                                          "weather_code": [weather_code],
                                          "temperature_max":[temp_max],
                                          "temperature_min":[temp_min]
                                       })
                              ], ignore_index=True)
                        
                    result_list.append({
                             "ID": row['ID'],
                             "latitude": row['latitude'],
                             "longitude": row['longitude'],
                             "startdate": row['startdate'] ,
                             "enddate": row["enddate"] ,
                             "weather_code": weather_code,
                             "temperature_max": temp_max,
                             "temperature_min": temp_min
                          })        

                except Exception as e:
                    error=f"Error fetching data:{e}"
                    self.errors = pd.concat([
                                         self.errors,
                                         pd.DataFrame({
                                                       "ID": [row["ID"]],
                                                       "Error": [error]
                                                      })
                                         ], ignore_index=True)   
                        
    def _get_existing(self, row, result_list):
            
        existing_entry = self.dataset[
            (self.dataset["latitude"] == row["latitude"]) &
            (self.dataset["longitude"] == row["longitude"]) &
            (self.dataset["startdate"] == row["startdate"]) &
            (self.dataset["enddate"] == row["enddate"])
            ]
			
        if not existing_entry.empty:
            weather_code = existing_entry.iloc[0]["weather_code"]
            temp_max = existing_entry.iloc[0]["temperature_max"]
            temp_min = existing_entry.iloc[0]["temperature_min"] 	    
            result_list.append({
                            "ID": row['ID'],
                            "latitude": row['latitude'],
                            "longitude": row['longitude'],
                            "startdate": row['startdate'] ,
                            "enddate": row["enddate"] ,
                            "weather_code": weather_code,
                            "temperature_max": temp_max,
                            "temperature_min": temp_min
                          })
            self.from_internal_db = self.from_internal_db + 1
        return existing_entry.empty
    
    def fetch_weather_data(self):
        """
        Fetch weather data for each row in the input DataFrame.
        DataFrame with columns ['ID', 'latitude', 'longitude', 'latitudedest', 'longitudedest','startdate',  'enddate',  'weather_code']
        """
        result_list = []

        for _, row in tqdm(self.input_dataframe.iterrows()):
            result_list1 = []
            result_list2 = []
            if self._has_any_error(row): continue
            
            row["startdate"]=  row["startdate"] if ((pd.to_datetime(row["enddate"], errors="coerce") > pd.to_datetime(row["startdate"])) or pd.isna(row["enddate"]) or row["enddate"] == "") else row["enddate"]
            row["enddate"] = row["enddate"] if pd.notna(pd.to_datetime(row["enddate"], errors="coerce")) and row["enddate"] != "" else row["startdate"]

            if self._get_existing(row,result_list1):
                    self._get_meteo_info(row,result_list1)  
                                     
            if row["latitudedest"] != "":
                if self._haversine(row["latitude"],row["longitude"],row["latitudedest"],row["longitudedest"]) > self.DIST_ROUTE_KM: 
                        row["latitude"] = row["latitudedest"]
                        row["longitude"] = row["longitudedest"]
                        if self._get_existing(row,result_list2):
                               self._get_meteo_info(row,result_list2)  
            
            # if result_list2: #si error??
            #     if result_list1[0]['weather_code'] > result_list2[0]['weather_code']: 
            #         result_list.append(result_list1[0])  
            #     else:
            #         result_list.append(result_list2[0])
            # else:
            #     result_list.append(result_list1[0]) 
                
            # result = max( filter(None, [result_list1[0] if result_list1 else None, result_list2[0] if result_list2 else None]),
            #                     key=lambda x: x["weather_code"],
            #                     default=None
            #                 )
            # if result is not None: result_list.append(result_list1[0])
            if result_list1 or result_list2:  # Verifica si alguna lista tiene valores
                if result_list1 and result_list2:  # Si ambas listas tienen valores
                    result = max(result_list1[0], result_list2[0], key=lambda x: x["weather_code"])
                else:  # Si solo una lista tiene valores
                    result = result_list1[0] if result_list1 else result_list2[0]

                result_list.append(result)  # Añade el mejor resultado a result_list
    
        self._save_dataset()

        result_dataframe = pd.DataFrame(result_list)
        print(result_dataframe.head())
        return result_dataframe

# Example:
# input_data = pd.DataFrame({
#     "ID": [1, 2, 3, 4, 5, 6 , 7,8],
#     "latitude": [52.52,"",48.85, 13.1550 ,12.7400,41.2808, 41.2808, 41.3879 ],
#     "longitude": [13.41,13.41, 2.35, 80.1960, 77.8200,1.9800, 1.9800, 2.16992],
#     "latitudedest": ["","","", "" ,"","","", 40.4168 ],
#     "longitudedest": ["","","", "" ,"","","",-3.7038],
#     "startdate": ["2025-01-25","" , "2025-01-20", "2020-08-17", "2020-08-17","2025-01-26","2025-01-26","2025-01-26" ],
#     "enddate": ["2025-01-21","2025-01-21", "2025-01-22", "2020-08-20", "2020-08-19","2025-01-28","2025-01-28",""]
# })
# meteo = MeteoData(input_data)
# result = meteo.fetch_weather_data()
# print(result)
# print("Weather: " , meteo.get_weather_code_info(int(result.loc[0]["weather_code"])))
# print("All Codes: ", meteo.get_weather_code_table())
# print("Errors:" , meteo.get_errors())