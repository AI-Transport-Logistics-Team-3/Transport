# MeteoData

## Descripción General

`MeteoData` es una clase en Python diseñada para gestionar y consultar información meteorológica utilizando la API de Open-Meteo. Esta clase permite realizar consultas tanto de datos meteorológicos actuales como históricos y almacenarlos en un dataset local para evitar llamadas redundantes a la API ya que la versión gratuita está limitada a 10.000 llamadas al día.

El dataset se almacena en un archivo CSV llamado `meteo_dataset.csv`.

---

## Requisitos

- Python 3.8 o superior
- Paquetes necesarios:
  - `openmeteo_requests`
  - `requests_cache`
  - `pandas`
  - `retry_requests`
  - `os`
  - `datetime`
  - `dateutil`
  - `math`

Puedes instalar estos paquetes ejecutando:

```bash
pip install openmeteo_requests requests_cache pandas retry_requests os datetime dateutil math
```

---
## Funcionalidad principal

La clase realiza las siguientes operaciones:
1. Recibe un dataframe de localizaciones de origen (longitud-latitud) y opcionalmente de destino, fechas de inicio y final de ruta.
2. Verifica si los datos meteorológicos de la localización de origen en las fechas indicadas ya están disponibles en un archivo CSV local (`meteo_dataset.csv`).
3. Si no se encuentran disponibles, realiza una consulta a la API de Open-Meteo.
4. Si se informa localización de destino y la distancia con la localización de origen es superior a 100Km se realiza la consulta de datos meteorológicos de la localización de destino.
5. Si la duración de la ruta es superior a un día se cogen los datos cuyo wmo code es mayor (clima más adverso).
6. Entre los datos de la localización de origen y destino también proporciona los datos con climatología más adversa.
7. Guarda las nuevas consultas en el archivo CSV para evitar futuras llamadas innecesarias a la API.
8. Devuelve un DataFrame con los resultados meteorológicos correspondientes.

## Características Principales

1. **Control de redundancia**:
   - Comprueba si los datos para una combinación de coordenadas, fecha y zona horaria ya existen en el dataset. Si existen, reutiliza esos datos.
2. **Consulta condicional**:
   - Realiza llamadas a la API de archivos históricos (`archive-api.open-meteo.com`) si la fecha es anterior a 3 meses, o a la API de pronósticos (`api.open-meteo.com`) que permite consultar entre los 3 meses anteriores y 16 días posteriores a la fecha actual.
2. **Almacenamiento de datos**:
   - Guarda los resultados de las consultas en un archivo CSV para futuras referencias.

## Entrada esperada
La clase `MeteoData` espera como entrada un DataFrame con las siguientes columnas:

| Columna         | Descripción                                                                                               | Valores esperados                  |
|-----------------|-----------------------------------------------------------------------------------------------------------|------------------------------------|
| `ID`           | Identificador único para cada registro.                                                                  | Enteros únicos.                    |
| `latitude`      | Latitud del punto de origen para la consulta meteorológica.                                              | Float entre -90 y 90.             |
| `longitude`     | Longitud del punto de origen para la consulta meteorológica.                                             | Float entre -180 y 180.           |
| `latitudedest`  | Latitud opcional del punto de destino, si es relevante para el análisis (puede ser ignorada por la clase).| Float entre -90 y 90 o vacío.     |
| `longitudedest` | Longitud opcional del punto de destino, si es relevante para el análisis (puede ser ignorada por la clase).| Float entre -180 y 180 o vacío.   |
| `startdate`     | Fecha de inicio del periodo para el que se solicitan datos meteorológicos.                                | String en formato `YYYY-MM-DD`.   |
| `enddate`       | Fecha de fin del periodo para el que se solicitan datos meteorológicos.                                   | String en formato `YYYY-MM-DD` o vacío (se usa `startdate` si no se proporciona). |

### Ejemplo de entrada
```python
import pandas as pd

input_data = pd.DataFrame({
    "ID": [1, 2],
    "latitude": [52.52, 48.85],
    "longitude": [13.41, 2.35],
    "latitudedest": [51.51, 40.73],
    "longitudedest": [-0.13, -73.93],
    "startdate": ["2025-01-20", "2023-12-15"],
    "enddate": ["2025-01-21", ""]
})
```

## Salida generada

El resultado de la clase será un DataFrame con las siguientes columnas:

| Columna         | Descripción                                                                                   | Valores esperados                                  |
|-----------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------|
| `ID`           | Identificador único del registro.                                                             | Enteros únicos.                                  |
| `latitude`      | Latitud del punto consultado.                                                                | Float entre -90 y 90.                           |
| `longitude`     | Longitud del punto consultado.                                                               | Float entre -180 y 180.                         |
| `latitudedest`  | Latitud del punto de destino, si es relevante.                                               | Float entre -90 y 90 o vacío.                   |
| `longitudedest` | Longitud del punto de destino, si es relevante.                                              | Float entre -180 y 180 o vacío.                 |
| `startdate`     | Fecha de inicio para los datos meteorológicos consultados.                                   | String en formato `YYYY-MM-DD`.                 |
| `enddate`       | Fecha de fin para los datos meteorológicos consultados.                                      | String en formato `YYYY-MM-DD` o vacío.         |
| `weather_code`  | Código WMO que representa el estado meteorológico del día.                                   | Enteros definidos por el estándar WMO.          |
| `temperature_max` | Temperatura máxima registrada o pronosticada en el periodo consultado.                     | Float en grados Celsius.                        |
| `temperature_min` | Temperatura mínima registrada o pronosticada en el periodo consultado.                     | Float en grados Celsius.                        |

El DataFrame incluye tanto los datos originales de entrada como las columnas adicionales con los resultados meteorológicos (weather_code, temperature_max, temperature_min).

## Uso

### Ejemplo de Uso

```python
import pandas as pd
from meteo_data import MeteoData

# Crear un DataFrame de entrada
input_data = pd.DataFrame({
    "ID": [1, 2],
    "latitude": [52.52, 48.85],
    "longitude": [13.41, 2.35],
    "Fecha": ["2025-01-20", "2020-12-31"],
    "timezone": ["Europe/Berlin", "Europe/Paris"]
})

# Instanciar la clase y obtener resultados
meteo = MeteoData(input_data)
result = meteo.fetch_weather_data()

print(result)
```

---

## Open-Meteo: Información General

[Open-Meteo](https://open-meteo.com/) es un servicio gratuito que proporciona datos meteorológicos de alta calidad mediante una API REST. Se centra en ofrecer datos de pronósticos y archivos históricos .

### API de Pronósticos (`https://api.open-meteo.com/v1/forecast`)

La API de pronósticos proporciona información detallada y personalizable sobre las condiciones meteorológicas actuales (hasta 3 meses anteriores) y futuras (16 días posteriores a la fecha actual). Entre las variables disponibles se incluyen:

- Código meteorológico (weather code - [WMO codes](https://www.nodc.noaa.gov/archive/arc0021/0002199/1.1/data/0-data/HTML/WMO-CODE/WMO4677.HTM))
- Temperatura máxima y mínima a 2 metros
- Índice UV máximo bajo cielo despejado
- Precipitación total y probabilidad de precipitación
- Viento: velocidad máxima, ráfagas y dirección dominante
- Nieve acumulada
- Radiación solar acumulada

### API Histórica (`https://archive-api.open-meteo.com/v1/archive`)

La API histórica permite acceder a datos meteorológicos pasados desde 1940. Esto resulta útil para analizar tendencias o realizar estudios meteorológicos retrospectivos. Las variables disponibles son las mismas a las de la API de pronósticos.

Ambas APIs permiten personalizar la consulta mediante parámetros como:
- Coordenadas geográficas (latitud y longitud)
- Fecha de inicio y fin
- Zona horaria


---



