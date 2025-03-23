import tkinter as tk
from tkinter import ttk, messagebox
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk
import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from tkcalendar import Calendar, DateEntry
from geopy.distance import geodesic
import re
from openai import OpenAI
import google.generativeai as genai
import openai
from NeuralNetwork.NN21_v4_single import *
from Meteo.meteo_forecast import MeteoData

DATASET ="\\NeuralNetwork\\clean_dataset_v2.csv"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


clientes = {
    23: {
        "nombre": "Otis elevator company (india) ltd",
        "direcciones": [
            {"direccion": "jamalpur, gurgaon, haryana", "latitud": "28.373519", "longitud": "76.835337", "estado": {"codigo": "11", "descripcion": "haryana"}},
            {"direccion": "kollur, medak, telangana", "latitud": "16.560192249175344", "longitud": "80.792293091599547", "estado": {"codigo": "16", "descripcion": "telangana"}},
        ],
    },
    19: {
        "nombre": "Larsen & toubro limited",
        "direcciones": [
            {"direccion": "kalyani, nadia, west bengal", "latitud": "22.95210237", "longitud": "88.4570148", "estado": {"codigo": "28", "descripcion": "west bengal"}},
        ],
    },
    4: {
        "nombre": "Comstar automotive technologies pvt ltd",
        "direcciones": [
            {"direccion": "shive, pune, maharashtra", "latitud": "18.750621", "longitud": "73.87719", "estado": {"codigo": "17", "descripcion": "maharashtra"}},
            
        ],
    },
}


fabricas = {
    1: {
        "nombre": "FABRICA BANGALORE",
        "direcciones": [
            {"direccion": "anekal, bangalore, karnataka", "latitud": "16.560192249175344", "longitud": "80.792293091599547", "estado": {"codigo": "9", "descripcion": "karnataka"}},
        ],
    },
    2: {
        "nombre": "FABRICA HOWRAH",
        "direcciones": [
            {"direccion": "kuldanga, howrah, west bengal", "latitud": "22.95210237429815", "longitud": "88.457014796076109", "estado": {"codigo": "18", "descripcion": "west bengal"}},
        ],
    },
}


proveedores = {
    1: {
        "codigo": "63-55471",
        "descripcion": "SUNITA CARRIERS PRIVATE LIMITED",
        "vehiculos": [
            {"codigo": "35", "descripcion": "40 FT 3XL Trailer 35MT"},
            {"codigo": "25", "descripcion": "28 FT Open Body 25MT"},
        ],
    },
    2: {
        "codigo": "65-55559",
        "descripcion": "DISTRIBUTION LOGISTICS INFRASTRUCTURE PRIVATE LTD",
        "vehiculos": [
            {"codigo": "14", "descripcion": "32 FT Multi-Axle 14MT - HCV"},
            {"codigo": "21", "descripcion": "24 / 26 FT Taurus Open 21MT - HCV"},
        ],
    },
}


cliente_seleccionado = None
direccion_seleccionada = None
latitud_seleccionada = 0.0
longitud_seleccionada = 0.0
estado_seleccionado = None

fabrica_seleccionada = None
direccion_fabrica_seleccionada = None
latitud_fabrica_seleccionada = 0.0
longitud_fabrica_seleccionada = 0.0
estado_fabrica_seleccionado = None

proveedor_seleccionado = None
vehiculos_disponibles = None

result = None
feature_importance = None


def seleccionar_cliente():
    global cliente_seleccionado, direccion_seleccionada, latitud_seleccionada, longitud_seleccionada, estado_seleccionado


    ventana_cliente = tk.Toplevel()
    ventana_cliente.title("Seleccionar Cliente")
    ventana_cliente.geometry("450x400")

    lbl_titulo = ttk.Label(ventana_cliente, text="Seleccione un cliente:", font=("Arial", 12))
    lbl_titulo.pack(pady=10)


    tree = ttk.Treeview(ventana_cliente, columns=("ID", "Nombre"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Nombre", text="Nombre")

    tree.column("ID", width=10, anchor="center")  
    tree.column("Nombre", width=250, anchor="w")  

    tree.pack(fill="both", expand=True, padx=10, pady=10)

    
    for cliente_id, datos in clientes.items():
        tree.insert("", "end", values=(cliente_id, datos["nombre"]))

    
    def seleccionar():
        global cliente_seleccionado, direccion_seleccionada, latitud_seleccionada, longitud_seleccionada, estado_seleccionado
        seleccion = tree.selection()
        if not seleccion:
            messagebox.showwarning("Error", "Por favor, seleccione un cliente.")
            return

        
        cliente_id = tree.item(seleccion[0], "values")[0]
        cliente_seleccionado = (cliente_id, clientes[int(cliente_id)]["nombre"])

        
        direcciones = clientes[int(cliente_id)]["direcciones"]
        if len(direcciones) == 1:
            direccion_seleccionada = direcciones[0]["direccion"]
            latitud_seleccionada = direcciones[0]["latitud"]
            longitud_seleccionada = direcciones[0]["longitud"]
            estado_seleccionado = f"{direcciones[0]['estado']['codigo']} - {direcciones[0]['estado']['descripcion']}"
            ventana_cliente.destroy()
            actualizar_campos_cliente()
        else:
            ventana_cliente.destroy()
            seleccionar_direccion(direcciones)

    btn_seleccionar = ttk.Button(ventana_cliente, text="Seleccionar", command=seleccionar)
    btn_seleccionar.pack(pady=10)

def seleccionar_direccion(direcciones):
    global direccion_seleccionada, estado_seleccionado, latitud_seleccionada, longitud_seleccionada

    ventana_direccion = tk.Toplevel()
    ventana_direccion.title("Seleccionar Dirección")
    ventana_direccion.geometry("800x400")

    lbl_titulo = ttk.Label(ventana_direccion, text="Seleccione una dirección", font=("Arial", 12))
    lbl_titulo.pack(pady=15)

    tree = ttk.Treeview(ventana_direccion, columns=("Dirección", "Estado", "Latitud", "Longitud"), show="headings")
    tree.heading("Dirección", text="Dirección")
    tree.heading("Estado", text="Estado")
    tree.heading("Latitud", text="Latitud")
    tree.heading("Longitud", text="Longitud")

    tree.column("Dirección", width=150, anchor="w")  
    tree.column("Estado", width=150, anchor="w") 
    tree.column("Latitud", width=50, anchor="w")  
    tree.column("Longitud", width=50, anchor="w")  

    tree.pack(fill="both", expand=True, padx=10, pady=10)

    for direccion in direcciones:
        tree.insert("", "end", values=(
            direccion["direccion"],
            f"{direccion['estado']['codigo']} - {direccion['estado']['descripcion']}",
            direccion["latitud"],
            direccion["longitud"]
        ))

    def seleccionar():
        global direccion_seleccionada,estado_seleccionado, latitud_seleccionada, longitud_seleccionada
        seleccion = tree.selection()
        if not seleccion:
            messagebox.showwarning("Error", "Por favor, seleccione una dirección.")
            return

        datos = tree.item(seleccion[0], "values")
        direccion_seleccionada = datos[0]
        latitud_seleccionada = datos[2]
        longitud_seleccionada = datos[3]
        estado_seleccionado = datos[1]
        ventana_direccion.destroy()
        actualizar_campos_cliente()

    btn_seleccionar = ttk.Button(ventana_direccion, text="Seleccionar", command=seleccionar)
    btn_seleccionar.pack(pady=15)

def seleccionar_fabrica():
    global fabrica_seleccionada, direccion_fabrica_seleccionada, latitud_fabrica_seleccionada, longitud_fabrica_seleccionada, estado_fabrica_seleccionado

    ventana_fabrica = tk.Toplevel()
    ventana_fabrica.title("Seleccionar Fábrica/Almacén")
    ventana_fabrica.geometry("450x400")

    lbl_titulo = ttk.Label(ventana_fabrica, text="Seleccione una fábrica/almacén:", font=("Arial", 12))
    lbl_titulo.pack(pady=10)

    tree = ttk.Treeview(ventana_fabrica, columns=("ID", "Nombre"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Nombre", text="Nombre")

    tree.column("ID", width=10, anchor="center")  
    tree.column("Nombre", width=150, anchor="w")  

    tree.pack(fill="both", expand=True, padx=10, pady=10)

    for fabrica_id, datos in fabricas.items():
        tree.insert("", "end", values=(fabrica_id, datos["nombre"]))


    def seleccionar():
        global fabrica_seleccionada, direccion_fabrica_seleccionada, latitud_fabrica_seleccionada, longitud_fabrica_seleccionada, estado_fabrica_seleccionado
        seleccion = tree.selection()
        if not seleccion:
            messagebox.showwarning("Error", "Por favor, seleccione una fábrica/almacén.")
            return


        fabrica_id = tree.item(seleccion[0], "values")[0]
        fabrica_seleccionada = (fabrica_id, fabricas[int(fabrica_id)]["nombre"])

        direcciones = fabricas[int(fabrica_id)]["direcciones"]
        if len(direcciones) == 1:
            direccion_fabrica_seleccionada = direcciones[0]["direccion"]
            latitud_fabrica_seleccionada = direcciones[0]["latitud"]
            longitud_fabrica_seleccionada = direcciones[0]["longitud"]
            estado_fabrica_seleccionado = f"{direcciones[0]['estado']['codigo']} - {direcciones[0]['estado']['descripcion']}"
            ventana_fabrica.destroy()
            actualizar_campos_fabrica()
        else:
            ventana_fabrica.destroy()
            seleccionar_direccion_fabrica(direcciones)

    btn_seleccionar = ttk.Button(ventana_fabrica, text="Seleccionar", command=seleccionar)
    btn_seleccionar.pack(pady=10)

def seleccionar_direccion_fabrica(direcciones):
    global direccion_fabrica_seleccionada, estado_fabrica_seleccionado, latitud_fabrica_seleccionada, longitud_fabrica_seleccionada

    ventana_direccion = tk.Toplevel()
    ventana_direccion.title("Seleccionar Dirección")
    ventana_direccion.geometry("500x400")

    lbl_titulo = ttk.Label(ventana_direccion, text="Seleccione una dirección:", font=("Arial", 12))
    lbl_titulo.pack(pady=15)

    tree = ttk.Treeview(ventana_direccion, columns=("Dirección", "Latitud", "Longitud", "Estado"), show="headings")
    tree.heading("Dirección", text="Dirección")
    tree.heading("Estado", text="Estado")
    tree.heading("Latitud", text="Latitud")
    tree.heading("Longitud", text="Longitud")
    

    tree.column("Dirección", width=150, anchor="w")  
    tree.column("Estado", width=150, anchor="w")  
    tree.column("Latitud", width=50, anchor="w")  
    tree.column("Longitud", width=50, anchor="w") 


    tree.pack(fill="both", expand=True, padx=10, pady=10)

    for direccion in direcciones:
        tree.insert("", "end", values=(
            direccion["direccion"],
            f"{direccion['estado']['codigo']} - {direccion['estado']['descripcion']}",
            direccion["latitud"],
            direccion["longitud"]
        ))

    def seleccionar():
        global direccion_fabrica_seleccionada, latitud_fabrica_seleccionada, longitud_fabrica_seleccionada, estado_fabrica_seleccionado
        seleccion = tree.selection()
        if not seleccion:
            messagebox.showwarning("Error", "Por favor, seleccione una dirección.")
            return

        datos = tree.item(seleccion[0], "values")
        direccion_fabrica_seleccionada = datos[0]
        latitud_fabrica_seleccionada = datos[2]
        longitud_fabrica_seleccionada = datos[3]
        estado_fabrica_seleccionado = datos[1]
        ventana_direccion.destroy()
        actualizar_campos_fabrica()

    btn_seleccionar = ttk.Button(ventana_direccion, text="Seleccionar", command=seleccionar)
    btn_seleccionar.pack(pady=15)

def seleccionar_proveedor():
    global proveedor_seleccionado, vehiculos_disponibles

    ventana_proveedor = tk.Toplevel()
    ventana_proveedor.title("Seleccionar Proveedor")
    ventana_proveedor.geometry("500x400")

    lbl_titulo = ttk.Label(ventana_proveedor, text="Seleccione un proveedor:", font=("Arial", 12))
    lbl_titulo.pack(pady=10)

    tree = ttk.Treeview(ventana_proveedor, columns=("ID", "Código", "Descripción"), show="headings")
    tree.heading("ID", text="ID")
    tree.heading("Código", text="Código")
    tree.heading("Descripción", text="Descripción")

    tree.column("ID", width=10, anchor="center")  
    tree.column("Código", width=50, anchor="w")  
    tree.column("Descripción", width=250, anchor="w")  

    tree.pack(fill="both", expand=True, padx=10, pady=10)


    for proveedor_id, datos in proveedores.items():
        tree.insert("", "end", values=(proveedor_id, datos["codigo"], datos["descripcion"]))

    def seleccionar():
        global proveedor_seleccionado, vehiculos_disponibles
        seleccion = tree.selection()
        if not seleccion:
            messagebox.showwarning("Error", "Por favor, seleccione un proveedor.")
            return

        proveedor_id = tree.item(seleccion[0], "values")[0]
        proveedor_seleccionado = (proveedor_id, proveedores[int(proveedor_id)]["codigo"], proveedores[int(proveedor_id)]["descripcion"])
        vehiculos_disponibles = proveedores[int(proveedor_id)]["vehiculos"]
        ventana_proveedor.destroy()
        actualizar_campos_proveedor()

    btn_seleccionar = ttk.Button(ventana_proveedor, text="Seleccionar", command=seleccionar)
    btn_seleccionar.pack(pady=10)

def actualizar_campos_cliente():
    global cliente_seleccionado, direccion_seleccionada, latitud_seleccionada, longitud_seleccionada, estado_seleccionado
    if cliente_seleccionado and direccion_seleccionada:
        entry_codigo_cliente.delete(0, tk.END)
        entry_codigo_cliente.insert(0, cliente_seleccionado[0])
        entry_nombre_cliente.delete(0, tk.END)
        entry_nombre_cliente.insert(0, cliente_seleccionado[1])
        entry_direccion_cliente.delete(0, tk.END)
        entry_direccion_cliente.insert(0, direccion_seleccionada)
 
def actualizar_campos_fabrica():
    global fabrica_seleccionada, direccion_fabrica_seleccionada, latitud_fabrica_seleccionada, longitud_fabrica_seleccionada, estado_fabrica_seleccionado
    if fabrica_seleccionada and direccion_fabrica_seleccionada:
        combo_origen.delete(0, tk.END)
        combo_origen.insert(0, fabrica_seleccionada[1])
      
def actualizar_campos_proveedor():
    global proveedor_seleccionado, vehiculos_disponibles
    if proveedor_seleccionado:
        entry_codigo_proveedor.delete(0, tk.END)
        entry_codigo_proveedor.insert(0, proveedor_seleccionado[1])
        entry_descripcion_proveedor.delete(0, tk.END)
        entry_descripcion_proveedor.insert(0, proveedor_seleccionado[2])
        if vehiculos_disponibles:
            combo_vehiculo['values'] = [f"{v['codigo']} - {v['descripcion']}" for v in vehiculos_disponibles]
            combo_vehiculo.current(0)


def evaluar_retraso_v2():
    def obtener_coordenadas_destino():
        lat = float(latitud_seleccionada)
        lon = float(longitud_seleccionada)
        return lat, lon

    def obtener_coordenadas_origen():
        lat = float(latitud_fabrica_seleccionada)
        lon = float(longitud_fabrica_seleccionada)
        return lat, lon

    def obtener_clima(org_lat, org_lon, lat, lon, fecha):
        
        input_data = pd.DataFrame({
            "ID": [1],
            "latitude": [org_lat],
            "longitude": [org_lon],
            "latitudedest": [lat ],
            "longitudedest": [lon],
            "startdate": [fecha],
            "enddate": [fecha]
        })
        meteo = MeteoData(input_data)
        result = meteo.fetch_weather_data()
        print(result)
        print("Weather: " , meteo.get_weather_code_info(int(result.loc[0]["weather_code"])))
        weather_code_info = meteo.get_weather_code_info(int(result.loc[0]["weather_code"]))
        weather_code = result.loc[0]["weather_code"]
        temperature_max =  result.loc[0]["temperature_max"]
        temperature_min = result.loc[0]["temperature_min"]

        return weather_code_info, weather_code, temperature_max, temperature_min


    def calcular_distancia(coords_origen, coords_destino):
        return geodesic(coords_origen, coords_destino).kilometers


    coords_origen = obtener_coordenadas_origen()
    coords_destino = obtener_coordenadas_destino()

    distancia = calcular_distancia(coords_origen, coords_destino)

    fecha_entrega = entry_fecha_entrega.get()
    weather_code_info, weather_code, temperature_max, temperature_min = obtener_clima(coords_origen[0],coords_origen[1], coords_destino[0], coords_destino[1], fecha_entrega[:10])
     
    fecha_entrega_dt = pd.to_datetime(fecha_entrega)
    planned_day_of_week = fecha_entrega_dt.dayofweek
    planned_hour = fecha_entrega_dt.hour
    planned_month = fecha_entrega_dt.month

    vehiculo = combo_vehiculo.get()
    cod_vehiculo = vehiculo[:2]

    match = re.search(r"(\d+)(?=\s*MT)", vehiculo)
    MT = match.group(1) if match else "0"

    mercancia = combo_mercancia.get()
    cod_mercancia = mercancia.split("-")[0].strip()

    cod_estado_orig = estado_fabrica_seleccionado.split("-")[0].strip()
    cod_estado_dest= estado_seleccionado.split("-")[0].strip()

    datos = pd.DataFrame({
        'distance': distancia,
        'Org_latitude': [coords_origen[0]],
        'Org_longitude': [coords_origen[1]],
        'Des_latitude': [coords_destino[0]],
        'Des_longitude': [coords_destino[1]],
        'MT': [MT],  
        'weather_code': [weather_code],
        'temperature_max': [temperature_max],
        'temperature_min': [temperature_min],
        'planned_day_of_week': [planned_day_of_week],
        'planned_hour': [planned_hour],
        'planned_month': [planned_month],
        'vehicleType_lbl': [cod_vehiculo],  
        'customerID_lbl': [int(entry_codigo_cliente.get())],  
        'supplierID_lbl': [entry_codigo_proveedor.get()[:2] ],
        'Material Shipped_lbl': [cod_mercancia],  
        'origin_state_lbl': [cod_estado_orig],  
        'dest_state_lbl': [cod_estado_dest], 
        'ontime_delay': [np.random.randint(0, 2,100)] #Deuda técnica. si lo cambio deja de funcionar 
    })


    single_input = datos.iloc[0][model_handler.feature_names].values
    
    
    delay,prob = model_handler.predict_single(single_input)
    res, shap_df = model_handler.predict_with_explanation(single_input)

    return delay  == 1, f"{prob}", weather_code_info, f"{distancia:.2f}", res, shap_df


def mostrar_explicabilidad():
    global result, feature_importance

    ventana_explicabilidad = tk.Toplevel(ventana)
    ventana_explicabilidad.title("Explicabilidad del Modelo")
    ventana_explicabilidad.geometry("800x650")
    ventana_explicabilidad.configure(bg="#f8f9fa")

    estilo = ThemedStyle(ventana_explicabilidad)
    estilo.set_theme("arc")
    
    feature_names = {
        'distance': 'Distancia',
        'Org_latitude': "Latitud de Origen",
        'Org_longitude': "Longitud de Origen",
        'Des_latitude': "Latitud de Destino",
        'Des_longitude': "Longitud de Destino",
        'MT': "MT",
        'weather_code': "Clima",
        'temperature_max': "Temperatura máxima",
        'temperature_min': "Temperatura mínima",
        'planned_day_of_week': "Día de la semana",
        'planned_hour': "Hora",
        'planned_month': "Mes",
        'vehicleType_lbl': "Tipo de Vehículo",
        'customerID_lbl': "Cliente",
        'supplierID_lbl': "Proveedor",
        'Material Shipped_lbl': "Material",
        'origin_state_lbl': "Estado de Origen",
        'dest_state_lbl': "Estado Destino"
    }

    ttk.Label(ventana_explicabilidad, text="Explicabilidad: Descripción de la predicción", font=("Arial", 12, "bold")).pack(pady=10, padx=15, anchor="center")
    ttk.Label(ventana_explicabilidad, text=lbl_retraso.cget("text"), font=("Arial", 10)).pack(pady=2, padx=15, anchor="w")
    ttk.Label(ventana_explicabilidad, text=f"{lbl_distancia.cget('text')}", font=("Arial", 10)).pack(pady=2, padx=15, anchor="w")
    ttk.Label(ventana_explicabilidad, text=f"{lbl_probabilidad_retraso.cget('text')}", font=("Arial", 10)).pack(pady=2, padx=15, anchor="w")

    ttk.Label(ventana_explicabilidad, text="Factores más influyentes:", font=("Arial", 12, "bold")).pack(pady=5, padx=15, anchor="w")

    frame_factores = ttk.Frame(ventana_explicabilidad)
    frame_factores.pack(pady=5, padx=15, fill="both", expand=True)

    texto_shap = "Resultados SHAP:\n"

    for idx, (_, row) in enumerate(feature_importance.iterrows()):
        nombre_traducido = feature_names.get(row['Feature'], row['Feature'])  
        valor_shap = f"{row['SHAP_Value']:.3f}"
        texto_shap += f"- {nombre_traducido}: {valor_shap}\n"

        columna = idx % 2  
        fila = idx // 2 

        ttk.Label(frame_factores, text=nombre_traducido, font=("Arial", 10)).grid(row=fila, column=columna * 2, padx=15, pady=2, sticky="w")
        ttk.Label(frame_factores, text=valor_shap, font=("Arial", 10, "bold")).grid(row=fila, column=columna * 2 + 1, padx=15, pady=2, sticky="w")

    ttk.Label(ventana_explicabilidad, text="Explicación del modelo:", font=("Arial", 12, "bold")).pack(pady=10, padx=10, anchor="w")

    text_explicacion = tk.Text(ventana_explicabilidad,font=("Arial", 10), wrap="word", height=10, width=100)
    text_explicacion.pack(padx=5, pady=5),

    def explicar_con_gpt(type="gemini"):

        prompt = f"Tenemos un modelo de red neuronal que predice la probabilidad de retraso en el transporte por carretera. \
                    En este caso, la predicción indica una probabilidad de retraso de {lbl_retraso.cget("text")}. \
                    El clima previsto para el día del envío es {lbl_probabilidad_retraso.cget('text')}, la fecha del envio será {entry_fecha_entrega.get()} \
                    y la distancia entre el origen y el destino es {lbl_distancia.cget('text')}. \
                    Tú eres un analista experto en transporte logístico y analista de datos y tu tarea es explicar breve y claramente los resultados obtenidos \
                    mediante el análisis SHAP para que el usuario del sistema pueda tomar decisiones informadas.\
                    La respuesta estará estructurada en los siguientes bloques: \
                    0. Descripción de la predicción:\
                    Si la probabilidad de retraso es alta, media o baja en base a la probabilidad del retraso, analizar la fecha, distancia y clima. \
                    1. Factores que incrementan la probabilidad de retraso:\
                    En este bloque,  explica las características que están aumentando la probabilidad de que se produzca un retraso en el transporte. \
                    Estos factores podrían estar relacionados con variables como el clima, temperaturas máximas o mínimas más extremas, la distancia, \
                    la congestión en las rutas, o cualquier otra condición \
                    que esté afectando negativamente el tiempo de entrega. Muestra análisis detallado sobre cómo cada uno de estos factores impacta en la predicción. \
                    2. Factores que reducen la probabilidad de retraso: \
                    En este bloque, explicar las características que están favoreciendo la puntualidad del transporte, es decir, \
                    aquellas que contribuyen a disminuir la probabilidad de que ocurra un retraso. \
                    Estos factores pueden incluir condiciones meteorológicas favorables, temperaturas mínimas o máximas más moderadas, una distancia más corta, \
                    o rutas más despejadas.   Aquí analiza cómo estas variables trabajan a favor para reducir el riesgo de retraso.\
                    3. **Sugerencias para disminuir la probabilidad de retraso:**\
                    Finalmente, en este bloque, ofrece algunas recomendaciones basadas en el análisis del modelo. \
                    Estos consejos están orientados a optimizar las condiciones del transporte y reducir la probabilidad de retraso.\
                    Las sugerencias pueden estar relacionadas con cambiar el horario de envío, elegir rutas alternativas o tomar decisiones \
                    sobre el clima para mitigar los efectos de condiciones adversas. Ten en cuenta que por necesidades del cliente es probable que haya fechas y\
                    otros parámetros que no se puedan modificar, ofrece alternativas realizas y viables.\
                    \nResultados SHAP a explicar: {texto_shap}. Debes utilizar un lenguaje claro y directo, evitando términos técnicos."


        if type == "OpenIA":
            client =  openai.OpenAI(api_key="TU_KEY")  
            respuesta = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )

            explicacion = respuesta.choices[0].message.content
        elif type == "gemini":
            
            genai.configure(api_key="TU_KEY")
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content(prompt)

            explicacion = response.text
        elif type == "deepseek":
            client = OpenAI(api_key="TU_KEY", base_url="https://api.deepseek.com")

            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello"},
                ],
                stream=False
            )

            explicacion = response.choices[0].message.content

        text_explicacion.delete("1.0", tk.END) 
        text_explicacion.insert(tk.END, explicacion)  

    ttk.Button(ventana_explicabilidad, text="Necesito más detalle (Explicar con IA)",  command=explicar_con_gpt).pack(pady=5)

    ttk.Button(ventana_explicabilidad, text="Cerrar", command=ventana_explicabilidad.destroy).pack(pady=5)



def guardar_pedido():
    messagebox.showinfo("Guardar Pedido", "Pedido guardado correctamente")

     

def obtener_fecha_entrega():
    global result, feature_importance

    retraso, probabilidad_retraso, clima, distancia, res, shap_df = evaluar_retraso_v2()
    result = res
    feature_importance = shap_df
    

    lbl_retraso.config(text=f"Probabilidad de retraso: {probabilidad_retraso}") 
    lbl_probabilidad_retraso.config(text=f"weather: {clima}")
    lbl_distancia.config(text=f"Distancia:{distancia}")
    btn_explicabilidad.grid()

    lbl_retraso.grid(row=15, column=1, columnspan=3, padx=15, pady=5,  sticky="w")
    lbl_probabilidad_retraso.grid(row=16, column=1, columnspan=3, padx=15, pady=5, sticky="w")
    lbl_distancia.grid(row=17, column=1, columnspan=3, padx=15, pady=5, sticky="w")

    lbl_retraso.config(font= ("Arial", 10, "bold"))
    lbl_probabilidad_retraso.config(font= ("Arial", 10))
    lbl_distancia.config(font= ("Arial", 10))

    if retraso == 1:
        lbl_retraso.config(foreground="red")  
        lbl_probabilidad_retraso.config(foreground="red")
        lbl_distancia.config(foreground="red")
    else:
        lbl_retraso.config(foreground="green")  
        lbl_probabilidad_retraso.config(foreground="green")
        lbl_distancia.config(foreground="green")


def seleccionar_fecha(entry=None):
    
    top = tk.Toplevel()
    top.title("Seleccionar Fecha")

    cal = Calendar(top, selectmode="day", year=2025, month=3, day=4, date_pattern="yyyy-mm-dd")
    cal.pack(pady=10, padx=10)

    frame_hora = ttk.Frame(top)
    frame_hora.pack(pady=5)

    horas = [f"{h:02d}" for h in range(24)]
    minutos = [f"{m:02d}" for m in range(0, 60, 5)]  

    ttk.Label(frame_hora, text="Hora:").pack(side="left", padx=5)
    cb_horas = ttk.Combobox(frame_hora, values=horas, width=3, state="readonly")
    cb_horas.set("10")  
    cb_horas.pack(side="left")

    ttk.Label(frame_hora, text=":").pack(side="left")

    cb_minutos = ttk.Combobox(frame_hora, values=minutos, width=3, state="readonly")
    cb_minutos.set("00")  
    cb_minutos.pack(side="left")

    def confirmar_fecha():
        fecha = cal.get_date()
        hora = cb_horas.get()
        minuto = cb_minutos.get()
        fecha_hora = f"{fecha} {hora}:{minuto}"
        entry.delete(0, tk.END)
        entry.insert(0, fecha_hora)
        top.destroy()

    btn_confirmar = ttk.Button(top, text="Confirmar", command=confirmar_fecha)
    btn_confirmar.pack(pady=10)


def calcular_fecha(entry=None):
    """Calcula automáticamente la fecha de entrega (2 días después de hoy)"""
    fecha_calculada = datetime.now() + timedelta(days=2)
    entry.delete(0, tk.END)
    entry.insert(0, fecha_calculada.strftime("%Y-%m-%d"))

def mostrar_info_cliente():
    ventana_info = tk.Toplevel()
    ventana_info.title("Información del Cliente")
    ventana_info.geometry("300x200")

    bg_color = ventana_info.cget("bg")  

    ttk.Label(ventana_info, text="📍 Ubicación:", font=("Arial", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
    lbl_latitud = tk.Label(ventana_info, text=f"🌍 Latitud: {latitud_seleccionada}", font=("Arial", 10), bg=bg_color)
    lbl_latitud.pack(anchor="w", padx=15)

    lbl_longitud = tk.Label(ventana_info, text=f"🗺️ Longitud: {longitud_seleccionada}", font=("Arial", 10), bg=bg_color)
    lbl_longitud.pack(anchor="w", padx=15)

    lbl_estado = tk.Label(ventana_info, text=f"🏛️ Estado: {estado_seleccionado}", font=("Arial", 10), bg=bg_color)
    lbl_estado.pack(anchor="w", padx=15)

def mostrar_info_origen():
    ventana_info = tk.Toplevel()
    ventana_info.title("Información de Origen mercancia")
    ventana_info.geometry("300x200")

    bg_color = ventana_info.cget("bg")  

    ttk.Label(ventana_info, text="📍Ubicación:", font=("Arial", 11, "bold")).pack(anchor="w", padx=10, pady=(10, 2))
    lbl_latitud = tk.Label(ventana_info, text=f"🌍 Latitud: {latitud_fabrica_seleccionada}", font=("Arial", 10), bg=bg_color)
    lbl_latitud.pack(anchor="w", padx=15)

    lbl_longitud = tk.Label(ventana_info, text=f"🗺️ Longitud: {longitud_fabrica_seleccionada}", font=("Arial", 10), bg=bg_color)
    lbl_longitud.pack(anchor="w", padx=15)

    lbl_estado = tk.Label(ventana_info, text=f"🏛️ Estado: {estado_fabrica_seleccionado}", font=("Arial", 10), bg=bg_color)
    lbl_estado.pack(anchor="w", padx=15)

# Ventana principal
ventana = tk.Tk()
ventana.title("Formulario de Pedido")
ventana.geometry("650x750")

estilo = ThemedStyle(ventana)
estilo.set_theme("arc")
estilo.configure("BotonAzul.TButton", font=("Arial", 11), padding=5, foreground="white")
estilo.map("BotonAzul.TButton", background=[("active", "#1e40af"), ("!disabled", "#1e3a8a")])
estilo.configure("TCombobox", selectbackground="white", selectforeground="black")

frame_principal = ttk.Frame(ventana)
frame_principal.pack(fill="both", expand=True, padx=20, pady=20)
canvas = tk.Canvas(frame_principal, bg="#f8f9fa")
scrollbar = ttk.Scrollbar(frame_principal, orient="vertical", command=canvas.yview)
contenedor = ttk.Frame(canvas, padding=10)
contenedor.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=contenedor, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

frame_logo = ttk.Frame(contenedor)
frame_logo.grid(row=0, column=0, columnspan=4, pady=10, sticky="ew")
img_logo = Image.open("C:/Sonia/ProyectoFinal/Transport/Screens/logo.png")  
img_logo = img_logo.resize((30, 30), Image.LANCZOS)
logo_tk = ImageTk.PhotoImage(img_logo)
lbl_logo = ttk.Label(frame_logo, image=logo_tk)
lbl_logo.grid(row=0, column=0, padx=10)
lbl_nombre = ttk.Label(frame_logo, text="Grupo 3 - Transportes y más", font=("Arial", 16, "bold"), foreground="#1e3a8a")
lbl_nombre.grid(row=0, column=1)


def crear_fila(etiqueta, fila, opciones=None, valor_default="", con_buscar=False,es_combobox=False, con_calendario=False, con_calculo=False, con_info=False, con_info_origen=False):
    lbl = ttk.Label(contenedor, text=etiqueta)
    lbl.grid(row=fila, column=0, padx=15, pady=5, sticky="w")

    frame_entrada = ttk.Frame(contenedor)
    frame_entrada.grid(row=fila, column=1, padx=15, pady=5, sticky="ew")
    if es_combobox:
        entrada = ttk.Combobox(frame_entrada, values=opciones, state="readonly", font=("Arial", 11), width=30)
        entrada.set(valor_default)
    elif opciones:
        entrada = ttk.Combobox(frame_entrada, values=opciones, state="readonly", font=("Arial", 11), width=30)
        entrada.set(valor_default)
    else:
        entrada = ttk.Entry(frame_entrada, font=("Arial", 11), width=30)
        entrada.insert(0, valor_default)
    entrada.pack(side="left", fill="x", expand=True)

    # Botón "Buscar" con icono
    if con_buscar:
        img_buscar = Image.open("/Screens/lupa.png")  
        img_buscar = img_buscar.resize((15, 15), Image.LANCZOS)
        icono_buscar = ImageTk.PhotoImage(img_buscar)
        if etiqueta == "Origen:":
            btn_buscar = ttk.Button(frame_entrada, image=icono_buscar, command=seleccionar_fabrica, width=14)
        elif etiqueta == "Código Proveedor:":
            btn_buscar = ttk.Button(frame_entrada, image=icono_buscar, command=seleccionar_proveedor,   width=14)
        else:
            btn_buscar = ttk.Button(frame_entrada, image=icono_buscar, command=seleccionar_cliente,  width=14)
        btn_buscar.image = icono_buscar
        btn_buscar.pack(side="left", padx=5)

    if con_calendario:
        btn_calendario = ttk.Button(frame_entrada, text="📅", width=3, command=lambda: seleccionar_fecha(entrada))
        btn_calendario.pack(side="left", padx=5)
    
    if con_calculo:
        btn_calcular = ttk.Button(frame_entrada, text="⏳", width=3, command=obtener_fecha_entrega) 
        btn_calcular.pack(side="left", padx=5)

    if con_info:
        btn_info = ttk.Button(frame_entrada, text=" ℹ️ ", command=mostrar_info_cliente,  width=3)
        btn_info.pack(side="left", padx=5)
    
    if con_info_origen:
        btn_info = ttk.Button(frame_entrada, text=" ℹ️ ", command=mostrar_info_origen,  width=3)
        btn_info.pack(side="left", padx=5)
    
    
    return entrada

entry_fecha_pedido = crear_fila("Fecha de pedido:", 1, valor_default="2024-02-26")
entry_booking_id = crear_fila("Booking ID:", 2, valor_default="MVCV0000927/082021")
combo_origen = crear_fila("Origen:", 3, valor_default="Seleccionar...", con_buscar=True, con_info_origen=True)

combo_mercancia = crear_fila("Mercancía:", 5, opciones=["432 - FXUWB-PLASTIC SEPERATER", "433 - FXUWB-PP SHEET", "1282 - WLT ASY RR DR WDO GARN MLDG","581 - INNER SYNCHRONIZER RING / 4. GANG","86 - AUTO PARTS","1091 - SPARE PARTS AUTOMOBILE", "721 - LU OIL RESERVOIR, POWER STEERING / ZF"], valor_default="Seleccionar...")
entry_codigo_cliente = crear_fila("Codigo Cliente:", 6, con_buscar=True)
entry_nombre_cliente = crear_fila("Cliente:", 7)
entry_direccion_cliente = crear_fila("Dirección Cliente:", 8, con_info=True)

entry_codigo_proveedor = crear_fila("Código Proveedor:", 10, con_buscar=True)
entry_descripcion_proveedor = crear_fila("Descripción Proveedor:", 11)

combo_vehiculo = crear_fila("Vehículo:", 12, es_combobox=True, valor_default="Seleccionar...")

entry_fecha_entrega = crear_fila("Fecha de Entrega:", 14, valor_default="", con_calendario=True, con_calculo=True)

lbl_fecha_entrega = ttk.Label(contenedor, text="Fecha de entrega: -", font=("Arial", 9))
lbl_retraso = ttk.Label(contenedor, text="Retraso: -", font=("Arial", 9))
lbl_probabilidad_retraso = ttk.Label(contenedor, text="Probabilidad de retraso: -", font=("Arial", 9))
lbl_distancia = ttk.Label(contenedor, text="Distancia: -", font=("Arial", 9))


btn_explicabilidad = ttk.Button(contenedor, text="Explícame la predicción", command=mostrar_explicabilidad)
btn_explicabilidad.grid(row=18, column=1, padx=10, pady=5, sticky="ew")
btn_explicabilidad.grid_remove() 


btn_cancelar = ttk.Button(contenedor, text="Cancelar", command=guardar_pedido)
btn_cancelar.grid(row=19, column=0, padx=10 ,sticky="ew")

btn_guardar = ttk.Button(contenedor, text="Guardar Pedido", command=guardar_pedido)
btn_guardar.grid(row=19, column=1, padx=10 ,sticky="ew")

#Carga el modelo 
df = pd.read_csv(DATASET)
model_handler = ModelHandler(df, train=False)

ventana.mainloop()