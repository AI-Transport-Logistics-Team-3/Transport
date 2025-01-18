from preprocessing.data_cleaner import data_preprocessing

# Crear una instancia de la clase de preprocesamiento
dp = data_preprocessing()

# Obtener el dataset procesado
df = dp.get_normaliced_dataset()

# Filtrar los datos según una condición
subset = df[df["supplierID_999"] == True]

# Obtener los datos originales correspondientes
original_data = dp.get_original_data(subset.index)