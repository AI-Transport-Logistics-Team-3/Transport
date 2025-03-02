# Delay Prediction Model

Contiene un conjunto de modelos de redes neuronales implementados en PyTorch para predecir retrasos en entregas basados en diversas características. Los modelos incluyen arquitecturas como redes neuronales simples, ResNet, DenseNet, Transformers y modelos híbridos.

## Descripción 

El objetivo de este proyecto es predecir si un envío llegará a tiempo o con retraso basándose en características como la distancia, ubicación, condiciones climáticas, tipo de vehículo, entre otras. Se utilizan diferentes arquitecturas de redes neuronales para comparar su rendimiento y seleccionar el mejor modelo.

## Estructura del Código

El código está organizado de la siguiente manera:

- **Modelos de Redes Neuronales**: Se definen varias arquitecturas de redes neuronales, incluyendo:
  - `DelayPredictorNN1`, `DelayPredictorNN2`, `DelayPredictorNN3`: Redes neuronales simples con diferentes configuraciones de capas.
  - `DelayPredictorWideNN`: Red neuronal con capas más anchas.
  - `DelayPredictorGatedNN`: Red neuronal con mecanismos de gating.
  - `DelayPredictorResNet`, `DelayPredictorResNet2`: Redes neuronales con bloques residuales.
  - `DelayPredictorDenseNet`: Red neuronal con conexiones densas.
  - `DelayPredictorTransformer`: Modelo basado en la arquitectura Transformer.
  - `DelayPredictorHybridNN`: Modelo híbrido que combina diferentes técnicas.

- **Entrenamiento y Evaluación**: La clase `ModelTrainer` se encarga de entrenar y evaluar los modelos. Incluye métodos para calcular métricas, graficar curvas ROC, matrices de confusión y explicar el modelo utilizando SHAP.

- **Ejecución de Modelos**: El script principal permite ejecutar diferentes modelos con hiperparámetros configurables y guardar los resultados en un archivo CSV.


## Arquitecturas de Redes Neuronales
### Redes Neuronales Simples (DelayPredictorNN1, DelayPredictorNN2, DelayPredictorNN3)
Estas redes neuronales tienen una estructura básica con varias capas fully connected (densas) y funciones de activación ReLU. La diferencia entre ellas radica en el número de capas y neuronas por capa.

- DelayPredictorNN1: 3 capas fully connected con 64, 32 y 16 neuronas respectivamente.

- DelayPredictorNN2: 4 capas fully connected con 128, 64, 32 y 16 neuronas.

- DelayPredictorNN3: 3 capas fully connected con 96, 48 y 24 neuronas.

### Red Neuronal Ancha (DelayPredictorWideNN)
Esta red neuronal tiene capas más anchas con 256 y 128 neuronas respectivamente. Esto permite capturar patrones más complejos en los datos.

###  Red Neuronal con Gating (DelayPredictorGatedNN)
Este modelo incorpora mecanismos de gating, similares a los utilizados en las redes LSTM, para controlar el flujo de información a través de las capas. Esto puede mejorar la capacidad del modelo para aprender dependencias a largo plazo.

###  Redes Residuales (DelayPredictorResNet, DelayPredictorResNet2)
Estas redes utilizan bloques residuales, que permiten que la información fluya directamente a través de las capas. Esto ayuda a mitigar el problema del vanishing gradient y permite entrenar redes más profundas.

- DelayPredictorResNet: Utiliza bloques residuales con 128 neuronas.

- DelayPredictorResNet2: Similar a la anterior pero con una estructura ligeramente diferente.

###  Red DenseNet (DelayPredictorDenseNet)
Este modelo utiliza conexiones densas, donde cada capa recibe la salida de todas las capas anteriores. Esto fomenta la reutilización de características y puede mejorar el rendimiento del modelo.

### Transformer (DelayPredictorTransformer)
Este modelo utiliza la arquitectura Transformer, que es especialmente efectiva para capturar dependencias a largo plazo en los datos. Incluye una capa de atención multi-head y normalización.

### Modelo Híbrido (DelayPredictorHybridNN)
Este modelo combina diferentes técnicas, como capas fully connected, gating y normalización, para aprovechar las ventajas de cada una.


## Requisitos

Para ejecutar el código, necesitas instalar las siguientes dependencias:

```bash
pip install torch numpy pandas scikit-learn shap matplotlib seaborn
```
## Uso

**Cómo Usar las Clases**
**1. Entrenar un Modelo**

Para entrenar un modelo, sigue estos pasos:
```python
from model import DelayPredictorNN1, ModelTrainer
import pandas as pd

# Cargar datos
df = pd.read_csv("ruta/al/archivo.csv")

# Inicializar el entrenador con el modelo deseado
trainer = ModelTrainer(df, DelayPredictorNN1, lr=0.001, weight_decay=0.00001, epochs=70, threshold=0.5)

# Entrenar el modelo
trainer.train()

# Evaluar el modelo
metrics = trainer.evaluate()
```

**2. Guardar y Cargar Pesos del Modelo**
Puedes guardar los pesos de un modelo entrenado para usarlos posteriormente:
```python
# Guardar pesos
trainer.model.save_model("ruta/al/modelo.pth")

# Cargar pesos
trainer.model.load_model("ruta/al/modelo.pth")
```

**3. Explicabilidad del Modelo con SHAP**
Para entender cómo el modelo toma sus decisiones, puedes usar SHAP:
```python
# Explicar el modelo
trainer.explain_model()
```

**4. Obtener Predicciones**
Puedes obtener predicciones del modelo entrenado:
```python
# Obtener predicciones
predictions = trainer.model.predict(trainer.X_val)
```

5. Analizar Resultados
Para analizar los resultados y obtener métricas detalladas:
```python
# Analizar resultados
results_df = trainer.analyze_results()
results_df.to_csv("resultados.csv", index=False)
```

