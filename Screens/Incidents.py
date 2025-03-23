import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Función para grabar incidencia
def grabar_incidencia():
    messagebox.showinfo("Grabar Incidencia", "Grabando incidencia...")

# Función para abrir la pantalla de consulta de incidencias
def consultar_incidencias():
    ventana_incidencias = tk.Toplevel()
    ventana_incidencias.title("Incidencias Registradas")
    ventana_incidencias.geometry("400x500")
    ventana_incidencias.configure(bg="#e3f2fd")

    # Botón para volver
    btn_volver = ttk.Button(ventana_incidencias, text="Volver", command=ventana_incidencias.destroy)
    btn_volver.pack(pady=10)

    # Título
    lbl_titulo = ttk.Label(ventana_incidencias, text="Incidencias Registradas", font=("Arial", 16))
    lbl_titulo.pack(pady=10)

    # Ejemplo de incidencia
    frame_incidencia = ttk.Frame(ventana_incidencias, padding=10)
    frame_incidencia.pack(fill="x", pady=5)

    lbl_audio = ttk.Label(frame_incidencia, text="Audio 1")
    lbl_audio.pack()

    lbl_transcripcion = ttk.Label(frame_incidencia, text="Transcripción: Tuve un problema con el motor en la autopista.", font=("Arial", 10))
    lbl_transcripcion.pack()

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Reportar Incidencia")
ventana.geometry("400x500")
ventana.configure(bg="#e3f2fd")

# Título
lbl_titulo = ttk.Label(ventana, text="Hola, Javier", font=("Arial", 24), background="#e3f2fd")
lbl_titulo.pack(pady=20)

# Botón para grabar incidencia
img_micro = Image.open("C:\Sonia\ProyectoFinal\Transport\Screens\microphone.png").resize((30, 30))  # Asegúrate de tener una imagen de micrófono
img_micro = ImageTk.PhotoImage(img_micro)
btn_grabar = ttk.Button(ventana, text="Grabar Incidencia", image=img_micro, compound="left", command=grabar_incidencia)
btn_grabar.pack(pady=20, ipadx=10, ipady=10)

# Botón para consultar incidencias
img_lista = Image.open("C:\Sonia\ProyectoFinal\Transport\Screens\list.png").resize((30, 30))  # Asegúrate de tener una imagen de lista
img_lista = ImageTk.PhotoImage(img_lista)
btn_consultar = ttk.Button(ventana, text="Consultar Incidencias", image=img_lista, compound="left", command=consultar_incidencias)
btn_consultar.pack(pady=20, ipadx=10, ipady=10)

# Ejecutar la aplicación
ventana.mainloop()