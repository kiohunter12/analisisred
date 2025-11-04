# scripts/modelo_pobreza.py
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def crear_modelo(input_dim, lr=0.001, dropout_rate=0.2, l2_reg=0.001):
    """
    Crea y compila una red neuronal para predecir pobreza.
    """
    modelo = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # salida entre 0 y 1 → umbral de pobreza
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return modelo
