
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.data_cleaner import data_preprocessing

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT = "output"

# dp = data_preprocessing()

# dp.save_normaliced_dataset()

# df = dp.get_normaliced_dataset()

# exit()


df = pd.read_csv("output/original.csv")

df["square_line_distance"] = (df["Org_latitude"] - df["Des_latitude"]) **2  + (df["Org_longitude"] - df["Des_longitude"]) **2

if False:

    mc = df.corr()

    # Crear el mapa de calor con seaborn
    # plt.figure(figsize=(40, 30))
    # sns.heatmap(mc, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    # plt.title("Matriz de Correlación")
    # plt.savefig(os.path.join(OUTPUT, "correlation_matrix.png"))

    print(mc[(mc["delay"].abs() > 0.2) & (mc["delay"].abs() < 0.95) ]["delay"] )

if False:
    # Configurar el gráfico 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar puntos en 3D con colores
    sc = ax.scatter(0.293127*df["vehicleType_40_FT_3XL_Trailer_35MT"], -0.237145*df["customerID_DMREXCHEUX"], 0.211092*df["square_line_distance"], c=df["delay"], cmap='viridis')

    # Agregar barra de color
    cb = plt.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label("Color")

    # Etiquetas de los ejes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Gráfico Espacial 3D")
    plt.show()


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar puntos en 3D con colores
    sc = ax.scatter(df["Org_latitude"], df["Org_longitude"], df["square_line_distance"], c=df["delay"], cmap='viridis')

    # Agregar barra de color
    cb = plt.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label("Color")

    # Etiquetas de los ejes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Gráfico Espacial 3D")
    plt.show()


    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar puntos en 3D con colores
    sc = ax.scatter(df["Des_latitude"], df["Des_longitude"], df["square_line_distance"], c=df["delay"], cmap='viridis')

    # Agregar barra de color
    cb = plt.colorbar(sc, ax=ax, pad=0.1)
    cb.set_label("Color")

    # Etiquetas de los ejes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Gráfico Espacial 3D")
    plt.show()

