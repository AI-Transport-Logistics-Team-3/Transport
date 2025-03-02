"""
Delay Prediction Model
----------------------

Este script contiene la implementación de varios modelos de redes neuronales para predecir retrasos en entregas.
Incluye arquitecturas como redes neuronales simples, ResNet, DenseNet, Transformers y modelos híbridos.

Autor: Sonia Sánchez
Fecha: 02/03/2025
Versión: 1.0

Descripción:
El objetivo de este proyecto es predecir si un envío llegará a tiempo o con retraso basándose en características
como la distancia, ubicación, condiciones climáticas, tipo de vehículo, entre otras. Se utilizan diferentes
arquitecturas de redes neuronales para comparar su rendimiento y seleccionar el mejor modelo.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shap
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, balanced_accuracy_score, cohen_kappa_score
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import itertools

#OPTIMAL_THRESHOLD =  0.3710 #0.3028#0.3713
CONF = "BestModels" #LoadModel, AutoML, BestModels
DATASET = "C:\Sonia\ProyectoFinal\Transport\Neural Network\clean_dataset.csv"
WEIGHT_PATH = "C:\\Sonia\\ProyectoFinal\\Transport\\enhanced_nn_model.pth"
AUTOML_RESULTS = "C:\\Sonia\\ProyectoFinal\\Transport\\output\\miniAutoML_results.csv"

class DelayPredictorNN1(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictorNN1, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=16)
       
        self.out_delay = nn.Linear(in_features=16, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.out_delay(x)  

class DelayPredictorNN2(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictorNN2, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=16)
        self.out_delay = nn.Linear(in_features=16, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.out_delay(x)  

class DelayPredictorNN3(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictorNN3, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=96)
        self.fc2 = nn.Linear(in_features=96, out_features=48)
        self.fc3 = nn.Linear(in_features=48, out_features=24)
        self.out_delay = nn.Linear(in_features=24, out_features=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)  

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return self.out_delay(x)

class DelayPredictorWideNN(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictorWideNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.out_delay = nn.Linear(in_features=128, out_features=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.out_delay(x)

class DelayPredictorGatedNN(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictorGatedNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.gate1 = nn.Linear(input_dim, 64)  
        self.fc2 = nn.Linear(64, 32)
        self.gate2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.gate3 = nn.Linear(32, 16)
        self.out_delay = nn.Linear(16, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x1 = self.relu(self.fc1(x)) * torch.sigmoid(self.gate1(x))
        x1 = self.dropout(x1)
        x2 = self.relu(self.fc2(x1)) * torch.sigmoid(self.gate2(x1))
        x2 = self.dropout(x2)
        x3 = self.relu(self.fc3(x2)) * torch.sigmoid(self.gate3(x2))
        x3 = self.dropout(x3)
        out = self.out_delay(x3)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, in_features)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.layer(x))  

class DelayPredictorResNet(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictorResNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

        self.residual_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.out_delay = nn.Linear(16, 1)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.residual_blocks(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.out_delay(x).view(-1, 1) 

class DelayPredictorResNet2(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictorResNet2, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.out_delay = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x_res = x  
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = x + self.fc1(x_res)  
        x_res = x
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = x + self.fc2(x_res)  
        x_res = x
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = x + self.fc3(x_res)  

        return self.out_delay(x)

class DelayPredictorDenseNet(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictorDenseNet, self).__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64 + input_dim, 32)
        self.fc3 = nn.Linear(32 + 64 + input_dim, 16)

        self.out_delay = nn.Linear(16 + 32 + 64 + input_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x1 = self.relu(self.fc1(x))
        x1 = self.dropout(x1)
        x2 = self.relu(self.fc2(torch.cat([x, x1], dim=1)))  
        x2 = self.dropout(x2)
        x3 = self.relu(self.fc3(torch.cat([x, x1, x2], dim=1)))  
        x3 = self.dropout(x3)

        out = self.out_delay(torch.cat([x, x1, x2, x3], dim=1))  
        return out


class DelayPredictorTransformer(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=64):
        super(DelayPredictorTransformer, self).__init__()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out_delay = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.input_layer(x).unsqueeze(1)  
        x, _ = self.self_attention(x, x, x)
        x = self.norm(x)  
        x = x.squeeze(1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        return self.out_delay(x)


class DelayPredictorHybridNN(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictorHybridNN, self).__init__()

        self.proj1 = nn.Linear(input_dim, 128)  
        self.proj2 = nn.Linear(128, 64)  
        self.proj3 = nn.Linear(64, 32)
        self.fc1 = nn.Linear(input_dim, 128)
        self.gate1 = nn.Linear(input_dim, 128)  
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.gate2 = nn.Linear(128, 64)  
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.gate3 = nn.Linear(64, 32)  
        self.bn3 = nn.BatchNorm1d(32)
        self.out_delay = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x_res = self.proj1(x)  
        x = self.relu(self.bn1(self.fc1(x))) * torch.sigmoid(self.gate1(x))  # x ahora tiene la dimensión correcta
        x = self.dropout(x) + x_res  
        x_res = self.proj2(x)  
        x = self.relu(self.bn2(self.fc2(x))) * torch.sigmoid(self.gate2(x))  
        x = self.dropout(x) + x_res  
        x_res = self.proj3(x)  
        x = self.relu(self.bn3(self.fc3(x))) * torch.sigmoid(self.gate3(x))  
        x = self.dropout(x) + x_res  

        return self.out_delay(x)


class DelayPredictorModelWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, epochs=70, lr=0.001, weight_decay = 1e-5, threshold = 0.5):
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.BCEWithLogitsLoss()
        self.threshold = threshold
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        print(self.epochs)
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()

            if epoch % 5 == 0:
                preds = (torch.sigmoid(outputs) >= self.threshold).float()
                acc = accuracy_score(y_tensor.numpy(), preds.numpy())
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, Accuracy={acc:.4f}")
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X_tensor)
            return (torch.sigmoid(outputs).numpy() >= self.threshold).astype(int)
        
    def predict_probability(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(X_tensor)
            return torch.sigmoid(outputs).numpy() 
         
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

class ModelTrainer:
    def __init__(self, df, model_class, weight_path="", lr=0.001, weight_decay = 0.00001, epochs = 70, threshold=0.5):
        print("Distribución de clases:", Counter(df['ontime_delay']))

        self.scaler = RobustScaler()#StandardScaler()
        self.feature_names = ['distance', 'Org_latitude', 'Org_longitude', 'Des_latitude', 'Des_longitude', 'MT',
                         'weather_code', 'temperature_max', 'temperature_min', 'planned_day_of_week',
                         'actual_day_of_week', 'planned_hour', 'actual_hour', 'planned_month', 'actual_month','vehicleType', 'customerID', 'supplierID', 'Material Shipped', 'origin_state', 'dest_state']

        self.df_original = df.copy()
        self.X = self.scaler.fit_transform(df[self.feature_names])
        self.y_delay = df['ontime_delay'].values.reshape(-1, 1)
        

        self.X_train, self.X_val, self.y_train_delay, self.y_val_delay = train_test_split(
            self.X, self.y_delay, test_size=0.3, random_state=42)

        base_model = model_class(input_dim=self.X_train.shape[1])
        self.model = DelayPredictorModelWrapper(base_model, lr=lr, weight_decay = weight_decay, epochs = epochs, threshold=threshold)

        if weight_path and os.path.exists(weight_path):
            self.model.load_model(weight_path)
            print(f"Model weights loaded from {weight_path}")


    def train(self):
        self.model.fit(self.X_train, self.y_train_delay)

    def evaluate(self):
        predictions = self.model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val_delay, predictions)
        print("============================================")
        print(self.model.model.__class__.__name__)
        print("============================================")
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(classification_report(self.y_val_delay, predictions))
        self.explain_model()
        importance_delay = self.feature_importance_permutation()
        self.plot_confusion_matrix(self.y_val_delay, predictions)
        self.plot_roc_curve(self.y_val_delay, self.model.predict_probability(self.X_val))
        return self.calculate_metrics(self.y_val_delay, predictions)
        
    def calculate_metrics(self, y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tnr = tn / (tn + fp)  # Specificity or True Negative Rate
        ppv = tp / (tp + fp)  # Precision or Positive Predictive Value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0 # Negative Predictive Value
        recall = tp / (tp + fn) # Recall or True Positive Rate
        accuracy= accuracy_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        n = fp + fn
        if n > 0:
            chi2, p = chi2_contingency([[fp, fn], [fn, fp]])[:2]  # Use chi2_contingency for correct p-value
            print(f"\nMcNemar's Test: Chi-squared = {chi2:.4f}, p-value = {p:.4f}")
        else:
            print("\nMcNemar's Test: Not applicable (no discordant pairs).")

        fpr, tpr, thresholds  = roc_curve(y_true, self.model.predict_probability(self.X_val))
        roc_auc = auc(fpr, tpr)
        youden_index = tpr - fpr
        optimal_threshold_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_threshold_idx]

        print("\nAdditional Metrics:")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}") 
        print(f"Recall (Sensitivity): {recall:.4f}") 
        print(f"Specificity: {tnr:.4f}")
        print(f"PPV (Precision): {ppv:.4f}")
        print(f"NPV: {npv:.4f}") 
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Kappa Index: {kappa:.4f}")

        return {
            'Model': self.model.model.__class__.__name__,
            'Accuracy': accuracy,
            'Balanced Accuracy': balanced_accuracy,
            'Kappa': kappa,
            'Recall':recall,
            'Specificity':tnr,
            'PPV':ppv,
            'NPV':npv,
            'Chi-squared':chi2,
            'p-value':p,
            'auc': roc_auc,
            'optimal_threshold_idx':optimal_threshold_idx,
            'optimal_threshold':optimal_threshold,
            'threshold': self.model.threshold,
            'HyperP - Learning rate ': self.model.lr,
            'HyperP - weight Decay ': self.model.weight_decay,
            'HyperP - epochs ': self.model.epochs
        }


    def plot_roc_curve(self, y_true, y_prob):
        fpr, tpr, thresholds  = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        youden_index = tpr - fpr
        optimal_threshold_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_threshold_idx]

        print(f"Umbral óptimo (Índice de Youden): {optimal_threshold:.4f}")
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title('Receiver Operating Characteristic')
        plt.title(self.model.model.__class__.__name__)
        plt.legend(loc="lower right")
        plt.show()
        optimal_predictions = (y_prob >= optimal_threshold).astype(int)
        accuracy = accuracy_score(y_true, optimal_predictions)
        print(f"Precisión con umbral óptimo: {accuracy:.4f}")
        print(classification_report(y_true, optimal_predictions))
        self.plot_confusion_matrix(y_true, optimal_predictions)
        print(f"AUC: {roc_auc},optimal_threshold_idx:{optimal_threshold_idx}, optimal_threshold: {optimal_threshold} ")

    def return_predictions(self):
        predictions = self.model.predict_probability(self.X_val)
        accuracy = accuracy_score(self.y_val_delay, predictions)
        print(f"Validation Accuracy: {accuracy:.4f}")

    def analyze_results(self):
        probabilities = self.model.predict_probability(self.X)

        result_df = self.df_original.copy()
        for i, col in enumerate(self.feature_names):
            result_df[f'Norm_{col}'] = self.X[:, i]

        result_df['predicted_probability'] = probabilities.flatten()
        result_df['actual_delay'] = self.y_delay.flatten()

        return result_df
    

    def explain_model(self):
        X_train_tensor = torch.tensor(self.X, dtype=torch.float32)
        X_val_tensor = torch.tensor(self.X_val, dtype=torch.float32)

        explainer = shap.Explainer(self.model.predict, X_train_tensor.numpy())

        shap_values = explainer(X_val_tensor.numpy())
        shap.summary_plot(shap_values, X_val_tensor.numpy(), feature_names=self.feature_names)
        shap.plots.bar(shap.Explanation(values=shap_values,
                                          data=X_val_tensor.numpy(), feature_names=self.feature_names)) 
        
    def feature_importance_permutation(self):
        result = permutation_importance(self.model, self.X_val, self.y_val_delay, 
                                        n_repeats=10, random_state=42, scoring="r2")

        importance = result.importances_mean
        sorted_idx = np.argsort(importance)[::-1]

        print("\nFeature Importance:")
        for i in sorted_idx:
            print(f"{self.feature_names[i]}: {importance[i]:.4f}")

        return importance
    
    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Delay', 'Delay'], yticklabels=['No Delay', 'Delay'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        #plt.title('Confusion Matrix')
        plt.title(self.model.model.__class__.__name__)
        plt.show()


def run_model(df, model_class, lr=0.001, weight_decay = 0.00001, epochs = 70, threshold = 0.5 ):
    trainer = ModelTrainer(df, model_class, lr=lr, weight_decay = weight_decay, epochs = epochs, threshold=threshold)
    trainer.train()
    results.append(trainer.evaluate())

def show_results():
    results_df = pd.DataFrame(results)
    calculate_score(results_df)
    best_models(results_df)
    print(results_df)

    results_df.to_csv(AUTOML_RESULTS)


def calculate_score(df):
    df["Recall_norm"] = (df["Recall"] - df["Recall"].min()) / (df["Recall"].max() - df["Recall"].min())
    df["BalancedAcc_norm"] = (df["Balanced Accuracy"] - df["Balanced Accuracy"].min()) / (df["Balanced Accuracy"].max() - df["Balanced Accuracy"].min())
    df["AUC_norm"] = (df["auc"] - df["auc"].min()) / (df["auc"].max() - df["auc"].min())
    # Ponderación: Recall (50%), Balanced Accuracy (30%), AUC (20%)
    df["Score"] = (df["Recall_norm"] * 0.5) + (df["BalancedAcc_norm"] * 0.3) + (df["AUC_norm"] * 0.2)

def best_models(df):

    df_bestmodel = df.groupby("Model").apply(lambda x: x.nlargest(2, "Score")).reset_index(drop=True)
    df_bestmodel.to_csv("C:\\Sonia\\ProyectoFinal\\Transport\\output\\best_models.csv", index=False)
  
    return 

df = pd.read_csv(DATASET)

results = []
if CONF=="LoadModel":
    trainer = ModelTrainer(df,WEIGHT_PATH)
elif CONF=="BestModels":	
    run_model(df, DelayPredictorNN1, lr=0.1, weight_decay = 0.001, epochs = 100, threshold=0.34)
    run_model(df, DelayPredictorNN2, lr=0.1, weight_decay = 0.0001, epochs = 100, threshold=0.5)
    run_model(df, DelayPredictorNN3, lr=0.01, weight_decay = 0.001, epochs = 100, threshold=0.371)
    run_model(df, DelayPredictorWideNN, lr=0.01, weight_decay = 0.000005, epochs = 100, threshold=0.34)
    run_model(df, DelayPredictorResNet, lr=0.01, weight_decay = 0.00001, epochs = 100, threshold=0.5)
    run_model(df,DelayPredictorResNet2, lr=0.01, weight_decay = 0.00001, epochs = 100, threshold=0.34)
    run_model(df,DelayPredictorDenseNet, lr=0.1, weight_decay = 0.000005, epochs = 100, threshold=0.44)
    run_model(df,DelayPredictorGatedNN, lr=0.1, weight_decay = 0.0001, epochs = 100, threshold=0.5)
    run_model(df, DelayPredictorTransformer, lr=0.01, weight_decay = 0.00001, epochs = 100, threshold=0.371)
    run_model(df, DelayPredictorHybridNN, lr=0.01, weight_decay = 0.00001, epochs = 100, threshold=0.34)
elif CONF=="AutoML":
    optimal_threshold = [0.34, 0.371, 0.44, 0.5]
    learning_rates = [0.0005, 0.001, 0.01, 0.1]
    weight_decays = [0.000005, 0.00001, 0.0001, 0.001]
    epochs = [20, 50, 70, 100]
    hparameters = list(itertools.product(learning_rates, weight_decays, epochs, optimal_threshold))
    for lr, weight_decay, epoch, threshold  in hparameters:
        run_model(df, DelayPredictorNN1, lr=lr, weight_decay = weight_decay, epochs = epoch, threshold=threshold)
        run_model(df, DelayPredictorNN2, lr=lr, weight_decay = weight_decay, epochs = epoch, threshold=threshold)
        run_model(df, DelayPredictorNN3, lr=lr, weight_decay = weight_decay, epochs = epoch, threshold=threshold)
        run_model(df, DelayPredictorWideNN, lr=lr, weight_decay = weight_decay, epochs = epoch, threshold=threshold)
        run_model(df, DelayPredictorResNet, lr=lr, weight_decay = weight_decay, epochs = epoch, threshold=threshold)
        run_model(df,DelayPredictorResNet2, lr=lr, weight_decay = weight_decay, epochs = epoch, threshold=threshold)
        run_model(df,DelayPredictorDenseNet, lr=lr, weight_decay = weight_decay, epochs = epoch, threshold=threshold)
        run_model(df,DelayPredictorGatedNN, lr=lr, weight_decay = weight_decay, epochs = epoch, threshold=threshold)
        run_model(df, DelayPredictorTransformer, lr=lr, weight_decay = weight_decay, epochs = epoch, threshold=threshold)
        run_model(df, DelayPredictorHybridNN, lr=lr, weight_decay = weight_decay, epochs = epoch, threshold=threshold)
else:
    print("Acción desconocida")
show_results()


#trainer.evaluate()
#trainer.explain_model()
#importance_delay = trainer.feature_importance_permutation()
#df_result = trainer.analyze_results()
#df_result.to_csv("C:\\Sonia\\ProyectoFinal\\Transport\\output\\results.csv")
#trainer.model.save_model("enhanced_nn_model.pth")
# trainer.model.load_model("enhanced_nn_model.pth")