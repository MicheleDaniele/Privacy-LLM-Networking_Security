import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping


# Caricamento del dataset dal file caricato in Colab
df = pd.read_excel('/content/sample_data/IWPC_2009.xlsx', sheet_name='IWPC 2009')

print("Colonne disponibili nel dataset:")
print(df.columns.tolist())

target_col = 'Therapeutic Dose of Warfarin'

X = df.drop(columns=[target_col])
y = df[target_col]

# Codifica delle Variabili Categoriche

# Converte variabili categoriche (es. "Gender") in colonne binarie (one-hot encoding).
X = pd.get_dummies(X, drop_first=True)

# Gestione dei Valori Mancanti

# Sostituisce i valori NaN con la media della colonna.
# Operazione necessaria per evitare errori durante l'addestramento.
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# Divisione in Training e Test Set

# Divide i dati in 70% training e 30% test.
# random_state=42 garantisce la riproducibilità dei risultati.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modello 1: Rete Neurale Profonda (overfitting)

# Architettura: 4 layer densi con attivazione ReLU.
# Overfitting: La profondità della rete (512 → 256 → 128) e l'assenza di regolarizzazione favoriscono l'overfitting.
model_overfit = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])

model_overfit.compile(optimizer='adam', loss='mse')
history_overfit = model_overfit.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    verbose=0
)

# Modello 2: Ridge Regression (No Overfitting)

# Ridge Regression: Aggiunge una penalità L2 ai pesi del modello per ridurre l'overfitting.
# alpha=100000.0: Forza della regolarizzazione (valore alto → maggior regolarizzazione).
ridge = Ridge(alpha=100000.0)
ridge.fit(X_train_scaled, y_train)

# Valutazione dei Modelli

y_train_pred_overfit = model_overfit.predict(X_train_scaled).flatten()
y_test_pred_overfit = model_overfit.predict(X_test_scaled).flatten()
mse_train_overfit = mean_squared_error(y_train, y_train_pred_overfit)
mse_test_overfit = mean_squared_error(y_test, y_test_pred_overfit)

y_train_pred_ridge = ridge.predict(X_train_scaled)
y_test_pred_ridge = ridge.predict(X_test_scaled)
mse_train_ridge = mean_squared_error(y_train, y_train_pred_ridge)
mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)

print("\n=== Modello OVERFITTING (Rete Neurale Profonda) ===")
print(f"MSE Train: {mse_train_overfit:.2f}")
print(f"MSE Test:  {mse_test_overfit:.2f}")

print("\n=== Modello NO OVERFITTING (Ridge) ===")
print(f"MSE Train: {mse_train_ridge:.2f}")
print(f"MSE Test:  {mse_test_ridge:.2f}")

# Visualizzazione dei Risultati

# Si può notare che nel modello che overfitta, la loss sul training set diminuisce, mentre quella sul test set aumenta.
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_overfit.history['loss'], label='Train Loss')
plt.plot(history_overfit.history['val_loss'], label='Test Loss')
plt.title('Learning Curve - Modello Overfitting')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Train', 'Test'], [mse_train_ridge, mse_test_ridge])
plt.title('MSE - Modello Ridge')
plt.ylabel('MSE')
plt.show()
