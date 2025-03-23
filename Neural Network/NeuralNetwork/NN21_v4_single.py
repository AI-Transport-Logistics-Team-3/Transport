
"""
DelayPredictorWideNN 
----------------------

Este script contiene la implementación ddel modelo Wide NN para predecir retrasos en entregas.

Autor: Sonia Sánchez
Fecha: 02/03/2025
Versión: 1.0

Descripción:
El objetivo de este proyecto es predecir si un envío llegará a tiempo o con retraso basándose en características
como la distancia, ubicación, condiciones climáticas, tipo de vehículo, entre otras. Se utilizan diferentes
arquitecturas de redes neuronales para comparar su rendimiento y seleccionar el mejor modelo.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import shap
import pandas as pd

DATASET ="\\NeuralNetwork\\clean_dataset_v2.csv"

class DelayPredictorWideNN(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictorWideNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.out(x)


class DelayPredictorModelWrapper:
    def __init__(self, model, threshold=0.371, lr=0.001, weight_decay=1e-5):
        self.model = model
        self.threshold = threshold
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train(self, X_train, y_train, epochs=50):
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)

        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
        
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    def predict_single(self, X_single):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_single, dtype=torch.float32).unsqueeze(0)  # Asegurar dimensión batch
            output = self.model(X_tensor)
            prob = torch.sigmoid(output).item()
            prediction = int(prob >= self.threshold)
            return prediction, f"{prob * 100:.2f}%"

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.model.eval()


class ModelHandler:
    def __init__(self, df, weight_path="model.pth", train=False):
        self.feature_names = [
            'distance', 'Org_latitude', 'Org_longitude', 'Des_latitude', 'Des_longitude', 'MT',
            'weather_code', 'temperature_max', 'temperature_min', 'planned_day_of_week',
            'planned_hour', 'planned_month', 'vehicleType_lbl', 'customerID_lbl',
            'supplierID_lbl', 'Material Shipped_lbl', 'origin_state_lbl', 'dest_state_lbl'
        ]
        self.scaler = RobustScaler()
        self.weight_path = weight_path

        X = self.scaler.fit_transform(df[self.feature_names])
        y = df['ontime_delay'].values.reshape(-1, 1)
        self.X = X

        if os.path.exists(weight_path) and not train:
            print(f"Loading model from {weight_path}")
            self.model = DelayPredictorWideNN(input_dim=X.shape[1])
            self.wrapper = DelayPredictorModelWrapper(self.model)
            self.wrapper.load_model(weight_path)
        else:
            print("Training new model...")
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model = DelayPredictorWideNN(input_dim=self.X_train.shape[1])
            self.wrapper = DelayPredictorModelWrapper(self.model)
            self.wrapper.train(self.X_train, self.y_train, epochs=50)
            self.wrapper.save_model(weight_path)
            print(f"Model saved to {weight_path}")

    def predict_single(self, X_new):
        
        X_scaled = self.scaler.transform(np.array(X_new).reshape(1, -1))
        print(self.scaler)
        return self.wrapper.predict_single(X_scaled)
    
    def predict_with_explanation(self, X_new):

        X_scaled = self.scaler.transform(np.array(X_new).reshape(1, -1))

        prediction, probability = self.wrapper.predict_single(X_scaled)

        def model_fn(input_array):
            input_tensor = torch.tensor(input_array, dtype=torch.float32)
            with torch.no_grad():
                return self.wrapper.model(input_tensor).numpy()

        explainer = shap.Explainer(model_fn, self.X)  
        shap_values = explainer(X_scaled)

        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Value': shap_values.values.flatten()
        }).sort_values(by='SHAP_Value', ascending=False)

        result = pd.DataFrame({'Prediction': [prediction], 'Probability': [probability]})
        return result, feature_importance


if __name__ == "__main__":
 
    df_test = pd.DataFrame({
        'distance': 1000,
        'Org_latitude': 16.56019225,
        'Org_longitude': 80.79229309,
        'Des_latitude': 22.95210237,
        'Des_longitude': 88.4570148,
        'MT': 35,
        'weather_code': 50,
        'temperature_max':33,
        'temperature_min': 26,
        'planned_day_of_week': 1,
        'planned_hour': 16,
        'planned_month': 6,
        'vehicleType_lbl':36,
        'customerID_lbl': 19,
        'supplierID_lbl': 63,
        'Material Shipped_lbl': 86,
        'origin_state_lbl': 18,
        'dest_state_lbl': 28,
        'ontime_delay': np.random.randint(0, 2, 100)
    })

    df = pd.read_csv(DATASET)
    model_handler = ModelHandler(df, train=False)

    # Probar predicción con una sola observación
    single_input = df_test.iloc[0][model_handler.feature_names].values
    result = model_handler.predict_single(single_input)
    print("Predicción para una observación:", result)

    result, shap_df = model_handler.predict_with_explanation(single_input)
    
    print(result)
    print(shap_df)
