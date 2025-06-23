import os
import pickle
import numpy as np
from cant_modos import *


snapshot_data_file = f"generando_matriz_inversa/elementos de prueba/datos_entrenamiento_1000.pkl"

if os.path.exists(snapshot_data_file):
    print(f"Cargando datos pre-calculados desde '{snapshot_data_file}'...")
    with open(snapshot_data_file, "rb") as f:
        # Cargar y desempaquetar la tupla que contiene ambos: parámetros y resultados
        training_set, results = pickle.load(f)
    print("Carga completada.")
else:
    exit

solutions, list_A_scipy, list_fv = zip(*results)
snapshots_matrix = np.array(solutions).T

print("SNAPSHOTS OBTENIDOS Y PROCESADOS")

V_svd, sigma, Wt = np.linalg.svd(snapshots_matrix, full_matrices=False)
print(f"Valores singulares de desplazamientos: {sigma}")
n_modos_u = contar_modos_por_energia(sigma, energia_deseada=0.99999)
print(f"Cantidad de modos para desplazamientos: {n_modos_u}")
V_reducida = V_svd[:, :n_modos_u]

print("Reduciendo matrices de rigidez...")
A_reducidas = [V_reducida.T @ A_sparse @ V_reducida for A_sparse in list_A_scipy]
f_reducidos = [V_reducida.T @ fv_i for fv_i in list_fv]

B_vectores = [np.linalg.inv(A_r_i).flatten() for A_r_i in A_reducidas]
B_matriz = np.array(B_vectores).T

Phi, Sigma_B, _ = np.linalg.svd(B_matriz, full_matrices=False)

# Imprime los valores singulares para entender su magnitud
print("Valores singulares de la matriz de rigidez (Sigma_B):", Sigma_B)

n_modos_B = contar_modos_por_energia(Sigma_B, energia_deseada=0.99999)
print(f"Cantidad de modos para matriz de rigidez: {n_modos_B}")
Phi_reducida = Phi[:, :n_modos_B]

Thetas = Phi_reducida.T @ B_matriz


# --- PREPARACIÓN DE DATOS PARA EL MODELO DE FUERZA ---

# Renombrar para mayor claridad
X_fuerza = np.array(training_set)
y_fuerza = np.array(f_reducidos)

# --- DIVISIÓN EN CONJUNTOS DE ENTRENAMIENTO Y VALIDACIÓN ---
from sklearn.model_selection import train_test_split

print("\nDividiendo los datos de fuerza en entrenamiento y validación (80/20)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_fuerza, 
    y_fuerza, 
    test_size=0.2,    # 20% de los datos para validación
    random_state=42   # Semilla para que la división sea siempre la misma
)

# --- ENTRENAMIENTO DEL MODELO DE REGRESIÓN LINEAL ---
from sklearn.linear_model import LinearRegression

print("\nEntrenando el modelo de Regresión Lineal para la fuerza...")
modelo_fuerza = LinearRegression()
# Entrenar SOLO con el conjunto de entrenamiento
modelo_fuerza.fit(X_train, y_train)
print("Entrenamiento del modelo de fuerza completado.")

# --- EVALUACIÓN DEL MODELO ---

# 1. Evaluación en el Conjunto de VALIDACIÓN (el más importante)
print("\n--- Evaluación en el Conjunto de Validación ---")
y_pred_val = modelo_fuerza.predict(X_val)

errores_relativos_val = []
for i in range(len(y_val)):
    y_verdadero = y_val[i]
    y_predecido = y_pred_val[i]
    
    norma_verdadero = np.linalg.norm(y_verdadero)
    if norma_verdadero > 1e-10:
        error_relativo = np.linalg.norm(y_verdadero - y_predecido) / norma_verdadero
        errores_relativos_val.append(error_relativo * 100)

if errores_relativos_val:
    error_promedio_val = np.mean(errores_relativos_val)
    print(f"Error relativo promedio en el conjunto de VALIDACIÓN: {error_promedio_val:.6f}%")

# 2. Evaluación en el Conjunto de ENTRENAMIENTO (para comparar)
print("\n--- Evaluación en el Conjunto de Entrenamiento ---")
y_pred_train = modelo_fuerza.predict(X_train)

errores_relativos_train = []
for i in range(len(y_train)):
    y_verdadero = y_train[i]
    y_predecido = y_pred_train[i]
    
    norma_verdadero = np.linalg.norm(y_verdadero)
    if norma_verdadero > 1e-10:
        error_relativo = np.linalg.norm(y_verdadero - y_predecido) / norma_verdadero
        errores_relativos_train.append(error_relativo * 100)

if errores_relativos_train:
    error_promedio_train = np.mean(errores_relativos_train)
    print(f"Error relativo promedio en el conjunto de ENTRENAMIENTO: {error_promedio_train:.6f}%")

# puntaje R² para ambos conjuntos
score_val = modelo_fuerza.score(X_val, y_val)
score_train = modelo_fuerza.score(X_train, y_train)
print(f"\nPuntaje R² en Validación: {score_val:.6f}")
print(f"Puntaje R² en Entrenamiento: {score_train:.6f}")