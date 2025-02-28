import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Cargar el dataset
df = pd.read_csv("output/original.csv")

# Crear una nueva característica
df["square_line_distance"] = (df["Org_latitude"] - df["Des_latitude"]) ** 2 + (df["Org_longitude"] - df["Des_longitude"]) ** 2

# Definir X e y
X = df[["Org_latitude", "Org_longitude", "square_line_distance"]].values
y = df["delay"].values

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Modelo k-NN
print("Entrenando k-NN...")
knn = KNeighborsClassifier(n_neighbors=5)  # Número de vecinos = 5
knn.fit(X_train_scaled, y_train)

# Predicción con k-NN
y_pred_knn = knn.predict(X_test_scaled)
print("Resultados de k-NN:")
print(classification_report(y_test, y_pred_knn))
print("Precisión k-NN:", accuracy_score(y_test, y_pred_knn))

# Matriz de confusión de k-NN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# 2. Modelo SVM
print("\nEntrenando SVM...")
kernels = ["linear", "poly", "rbf", "sigmoid"]

best_svm_model = None
best_accuracy = 0
best_kernel = None
best_y_pred_svm = None

for kernel in kernels:
    print(f"\nProbando kernel: {kernel}")
    svm = SVC(kernel=kernel, C=1, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Predicción con SVM
    y_pred_svm = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_svm)
    
    print(f"Precisión del kernel {kernel}: {accuracy:.4f}")
    print(classification_report(y_test, y_pred_svm))
    
    # Guardar el mejor modelo y su predicción
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_kernel = kernel
        best_svm_model = svm
        best_y_pred_svm = y_pred_svm

print(f"\nMejor kernel SVM: {best_kernel} con precisión: {best_accuracy:.4f}")

# Matriz de confusión del mejor SVM
conf_matrix_svm = confusion_matrix(y_test, best_y_pred_svm)

# Visualización de las matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Matriz de confusión de k-NN
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap="Blues", ax=axes[0])
axes[0].set_title("Matriz de Confusión - k-NN")
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Verdadero")

# Matriz de confusión del mejor SVM
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap="Greens", ax=axes[1])
axes[1].set_title(f"Matriz de Confusión - SVM (kernel={best_kernel})")
axes[1].set_xlabel("Predicción")
axes[1].set_ylabel("Verdadero")

plt.tight_layout()
plt.show()


print("-"*30)

df2 = df.drop(columns=["ontime", "Data_Ping_time"]).dropna(axis=1)
print(df2.columns)

X = df2.drop(columns=["delay"]).values
y = df2["delay"].values

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Modelo k-NN
print("Entrenando k-NN...")
knn = KNeighborsClassifier(n_neighbors=5)  # Número de vecinos = 5
knn.fit(X_train_scaled, y_train)

# Predicción con k-NN
y_pred_knn = knn.predict(X_test_scaled)
print("Resultados de k-NN:")
print(classification_report(y_test, y_pred_knn))
print("Precisión k-NN:", accuracy_score(y_test, y_pred_knn))

# Matriz de confusión de k-NN
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

# 2. Modelo SVM
print("\nEntrenando SVM...")
kernels = ["linear", "poly", "rbf", "sigmoid"]

best_svm_model = None
best_accuracy = 0
best_kernel = None
best_y_pred_svm = None

for kernel in kernels:
    print(f"\nProbando kernel: {kernel}")
    svm = SVC(kernel=kernel, C=1, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    # Predicción con SVM
    y_pred_svm = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_svm)
    
    print(f"Precisión del kernel {kernel}: {accuracy:.4f}")
    print(classification_report(y_test, y_pred_svm))
    
    # Guardar el mejor modelo y su predicción
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_kernel = kernel
        best_svm_model = svm
        best_y_pred_svm = y_pred_svm

print(f"\nMejor kernel SVM: {best_kernel} con precisión: {best_accuracy:.4f}")

# Matriz de confusión del mejor SVM
conf_matrix_svm = confusion_matrix(y_test, best_y_pred_svm)

# Visualización de las matrices de confusión
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Matriz de confusión de k-NN
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap="Blues", ax=axes[0])
axes[0].set_title("Matriz de Confusión - k-NN")
axes[0].set_xlabel("Predicción")
axes[0].set_ylabel("Verdadero")

# Matriz de confusión del mejor SVM
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap="Greens", ax=axes[1])
axes[1].set_title(f"Matriz de Confusión - SVM (kernel={best_kernel})")
axes[1].set_xlabel("Predicción")
axes[1].set_ylabel("Verdadero")

plt.tight_layout()
plt.show()