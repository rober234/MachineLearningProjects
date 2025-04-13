# conversor_celsius_fahrenheit.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 🔧 Forzar uso de CPU (opcional)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# Nombre del archivo del modelo guardado
MODELO_PATH = "models/modelo_entrenado.h5"

# Cargar modelo si ya está entrenado
if os.path.exists(MODELO_PATH):
    modelo = load_model(MODELO_PATH)
    print(" Modelo cargado desde archivo.")
else:
    print(" Entrenando modelo desde cero...")

    # Datos de entrenamiento: Celsius -> Fahrenheit
    celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
    fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

    # Definir el modelo
    modelo = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(units=1)
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_squared_error'
    )

    # Entrenar el modelo
    historial = modelo.fit(celsius, fahrenheit, epochs=600, verbose=0)

    # Guardar el modelo
    modelo.save(MODELO_PATH)
    print(f"Modelo entrenado y guardado en '{MODELO_PATH}'.")

    # Guardar gráfica de pérdida
    plt.figure(figsize=(8, 5))
    plt.plot(historial.history['loss'])
    plt.title("Evolución de la pérdida durante el entrenamiento")
    plt.xlabel("Época")
    plt.ylabel("Pérdida (MSE)")
    plt.grid(True)
    plt.savefig("images/loss_plot.png")
    print("Gráfica de pérdida guardada como 'loss_plot.png'.")

# Predicción
valor_input = 155.0
resultado = modelo.predict(np.array([[valor_input]]))
print(f" Predicción: {valor_input}°C son aproximadamente {resultado[0][0]:.2f}°F.")
