import os
import pickle
import time
import numpy as np
import pandas as pd

# --- Scikit-learn imports ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV # Importante para la búsqueda

# --- Constantes para formato de texto ---
C_BLUE_BOLD = "\033[1;37;44m"
C_GREEN_BOLD = "\033[1;32m"
C_RESET = "\033[0m"

# --- 1. Carga de Datos (Sin cambios) ---
training_file = "prueba_train_1000_chico.pkl"
testing_file = "prueba_test_30_chico.pkl"

try:
    with open(training_file, "rb") as f:
        reduced_basis, salidas, training_set, _ = pickle.load(f)
    with open(testing_file, "rb") as f:
        salidas_esperadas, testing_set, _ = pickle.load(f)
    print("✅ Datos cargados desde archivos .pkl.")
except FileNotFoundError:
    print("❌ Archivos .pkl no encontrados. Por favor, genera los datos primero.")
    exit()

X_train, y_train = np.array(training_set), np.array(salidas)
X_test, y_test = np.array(testing_set), np.array(salidas_esperadas)
V = np.array(reduced_basis)

# --- 2. Transformador Personalizado (Sin cambios) ---
class InverseFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, powers=[-1, -2, -3]): self.powers = powers
    def fit(self, X, y=None): return self
    def transform(self, X, y=None):
        X_inv = np.copy(X).astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            for p in self.powers: X_inv = np.c_[X_inv, np.power(X, p)]
        return np.nan_to_num(X_inv)

# --- 3. Función de Evaluación (Sin cambios) ---
def evaluar_y_mostrar_resultados(model, name, X_train, y_train, X_test, y_test, V):
    print(f"\n--- Evaluando Modelo Final: {name} ---")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    print(f"\n{C_BLUE_BOLD} Evaluación con Error Relativo (L2) {C_RESET}")
    errors_rel_test = []
    for i in range(len(y_test)):
        u_h, u_N = V @ y_test[i], V @ y_pred_test[i]
        errors_rel_test.append(np.linalg.norm(u_h - u_N) / np.linalg.norm(u_h))
    print("\nTESTEO")
    print(f"  Error Medio: {np.mean(errors_rel_test):.4%}")
    print(f"  Varianza:    {np.var(errors_rel_test):.4e}")
    print(f"  Error Máximo:  {np.max(errors_rel_test):.4%}")
    
    return {
        "Modelo": name,
        "Err. Relativo Medio (Test)": np.mean(errors_rel_test),
    }

# --- 4. Definición de Pipelines y Rejillas de Búsqueda ---
# Nos enfocaremos en los modelos que tienen hiperparámetros críticos que tunear.

# Pipeline para ElasticNet
pipe_elastic = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', ElasticNet(max_iter=5000)) # Aumentamos iteraciones
])

# Pipeline para el modelo Polinómico + Inverso
pipe_poly_inv = Pipeline([
    ('features', FeatureUnion([
        ('poly', PolynomialFeatures(include_bias=False)),
        ('inverse', InverseFeatures(powers=[-1, -2, -3]))
    ])),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(max_iter=5000)) # Usamos Ridge por su estabilidad
])

# Rejillas de parámetros a probar para cada modelo
# La sintaxis 'regressor__alpha' se refiere al parámetro 'alpha' del paso 'regressor' en el Pipeline
param_grid_elastic = {
    'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
    'regressor__l1_ratio': [0.1, 0.5, 0.9]
}

param_grid_poly_inv = {
    'features__poly__degree': [2, 3, 4], # Probaremos distintos grados polinómicos
    'regressor__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0] # Un rango amplio de alphas para la regularización
}

# --- 5. Búsqueda, Entrenamiento y Evaluación ---
final_results = []
# cv=5 significa validación cruzada de 5 particiones (un estándar robusto)
# scoring='neg_mean_squared_error' es una métrica estándar para regresión en GridSearchCV
# n_jobs=-1 usa todos los procesadores disponibles para acelerar la búsqueda

print("\n" + "="*60)
print("🚀 INICIANDO BÚSQUEDA DE HIPERPARÁMETROS CON GridSearchCV 🚀")
print("="*60)

# --- Búsqueda para ElasticNet ---
print(f"\n{'='*20} Modelo: ElasticNet {'='*20}")
search_elastic = GridSearchCV(pipe_elastic, param_grid_elastic, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
start_time = time.time()
search_elastic.fit(X_train, y_train)
duration = time.time() - start_time

print(f"⏱️  Tiempo de búsqueda: {duration:.3f} segundos")
print(f"{C_GREEN_BOLD}Mejores parámetros encontrados: {search_elastic.best_params_}{C_RESET}")
metrics = evaluar_y_mostrar_resultados(search_elastic.best_estimator_, "ElasticNet (Optimizado)", X_train, y_train, X_test, y_test, V)
metrics["Tiempo (s)"] = duration
metrics["Mejores Parámetros"] = search_elastic.best_params_
final_results.append(metrics)


# --- Búsqueda para Polinómico + Inverso ---
print(f"\n{'='*20} Modelo: Polinómico + Inverso {'='*20}")
search_poly = GridSearchCV(pipe_poly_inv, param_grid_poly_inv, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
start_time = time.time()
search_poly.fit(X_train, y_train)
duration = time.time() - start_time

print(f"⏱️  Tiempo de búsqueda: {duration:.3f} segundos")
print(f"{C_GREEN_BOLD}Mejores parámetros encontrados: {search_poly.best_params_}{C_RESET}")
metrics = evaluar_y_mostrar_resultados(search_poly.best_estimator_, "Polinómico + Inverso (Optimizado)", X_train, y_train, X_test, y_test, V)
metrics["Tiempo (s)"] = duration
metrics["Mejores Parámetros"] = search_poly.best_params_
final_results.append(metrics)


# --- 6. Resumen Final ---
results_df = pd.DataFrame(final_results)
results_df_sorted = results_df.sort_values(by="Err. Relativo Medio (Test)")

print("\n" + "="*80)
print("📊 RESUMEN COMPARATIVO DE MODELOS OPTIMIZADOS")
print("="*80)
print(results_df_sorted.to_string(index=False))