# Data Preprocessing

Este script realiza un preprocesamiento de datos para un archivo Excel, aplicando varias transformaciones como conversión de fechas, eliminación de columnas innecesarias, codificación de datos categóricos y normalización de columnas numéricas.

## Descripción del Proyecto

Este proyecto permite convertir datos de un archivo Excel a un formato adecuado para el análisis de datos, incluyendo:

- Conversión de fechas a formato Unix timestamp.
- Eliminación de columnas innecesarias.
- Separación de coordenadas geográficas en columnas de latitud y longitud.
- Codificación de variables categóricas a formato numérico.
- Normalización de columnas numéricas usando la normalización Z-score.

## Requisitos

Para ejecutar el script, necesitarás tener las siguientes dependencias instaladas:

- `pandas`
- `numpy`
- `openpyxl` (para leer archivos Excel)
- `kagglehub` (para descargar datasets de la base de datos de Kaggle)

Instalación de dependencias:

```bash
pip install pandas numpy openpyxl
```

Funcionalidades
1. _excel_to_csv()
Convierte un archivo Excel a un DataFrame de Pandas y lo guarda como archivo CSV si se especifica.

2. delete_unnecesary_information()
Elimina columnas innecesarias del DataFrame para mejorar el rendimiento.

3. change_to_datetime()
Convierte las columnas de fechas en el DataFrame a formato Unix timestamp.

4. change_longitude_altitude()
Separa las coordenadas geográficas de latitud y longitud en columnas separadas.

5. change_categoricall_to_numerical()
Convierte las columnas categóricas a valores numéricos usando varias técnicas de codificación.

6. normalice_dataset()
Normaliza las columnas numéricas utilizando la normalización Z-score.

7. save_preprocessed_dataset()
Guarda el DataFrame procesado en el formato deseado (csv o xlsx).

8. save_normaliced_dataset()
Guarda el DataFrame normalizado en el formato deseado (csv o xlsx).

9. save_original_dataset()
Guarda el DataFrame original en el formato deseado (csv o xlsx).

## Uso
Para ejecutar el script y realizar el preprocesamiento de datos, puedes utilizar el siguiente bloque de código:

```python
from data_cleaner import data_preprocessing

# Crear una instancia de la clase de preprocesamiento
dp = data_preprocessing()

# Obtener el dataset procesado
df = dp.get_normaliced_dataset()

# Filtrar los datos según una condición
subset = df[df["supplierID_999"] == True]

# Obtener los datos originales correspondientes
original_data = dp.get_original_data(subset.index)
```

## Importar la Clase desde Otro Archivo
Si deseas usar la clase data_preprocessing en otro archivo, puedes importarla de la siguiente manera:

```python
from preprocessing.data_cleaner import data_preprocessing

# Crear una instancia de la clase de preprocesamiento
dp = data_preprocessing()

# Utilizar las funciones de la clase según sea necesario
df = dp.get_normaliced_dataset()
```

Asegúrate de tener el archivo data_preprocessing.py en la misma carpeta o en una ruta accesible para importar la clase correctamente.

## Configuración de Archivos
Asegúrate de tener un archivo Excel llamado Delivery truck trip data.xlsx ubicado en la carpeta Data antes de ejecutar el script. Los resultados procesados se guardarán en la carpeta output.

## Estructura de Archivos
```
Data/
    └── Delivery truck trip data.xlsx
output/
    └── (Archivos procesados guardados)
```
