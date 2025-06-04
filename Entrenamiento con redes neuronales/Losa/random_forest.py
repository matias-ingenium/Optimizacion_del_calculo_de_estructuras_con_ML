import os
import pickle
import time
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# --- Constantes para colores (opcional) ---
GREEN = "\033[92m"
RESET = "\033[0m"
BLUE = "\033[94m"
YELLOW = "\033[93m"

# --- Definición de nombres de archivo ---
training_file = "prueba_train_4000_chico.pkl"
testing_file = "prueba_test_40_chico.pkl"

# --- Carga o generación de datos ---
if os.path.exists(training_file) and os.path.exists(testing_file):
    with open(training_file, "rb") as f:
        reduced_basis, salidas, training_set, snapshots_matrix = pickle.load(f)
    with open(testing_file, "rb") as f:
        salidas_esperadas, testing_set, result_matrix = pickle.load(f)
    print("Datos cargados desde archivo.")
else:
    print(f"{YELLOW}Archivos de datos no encontrados. Generando nuevos datos...{RESET}")
    # Asumimos que tienes estos módulos disponibles y funcionan
    try:
        from training_losa import training_data
        from testing_losa import testing_data
    except ImportError:
        print(f"{YELLOW}Error: No se pudieron importar 'training_losa' o 'testing_losa'.{RESET}")
        print("Asegúrate de que estos archivos estén en la misma carpeta o sean accesibles.")
        exit()

    print("Generando datos de entrenamiento...")
    start_time_gen = time.time()
    reduced_basis, salidas, training_set, snapshots_matrix = training_data(1000)
    duration_gen = time.time() - start_time_gen
    print(f"{GREEN}Duración de training_data: {duration_gen:.2f} segundos{RESET}")

    print("Generando datos de testeo...")
    start_time_gen = time.time()
    salidas_esperadas, testing_set, result_matrix = testing_data(30, reduced_basis)
    duration_gen = time.time() - start_time_gen
    print(f"{GREEN}Duración de testing_data: {duration_gen:.2f} segundos{RESET}")


    training_set = np.array(training_set)
    testing_set = np.array(testing_set)
    salidas = np.array(salidas)
    salidas_esperadas = np.array(salidas_esperadas)

    print("Guardando datos generados...")
    with open(training_file, "wb") as f:
        pickle.dump((reduced_basis, salidas, training_set, snapshots_matrix), f)
    with open(testing_file, "wb") as f:
        pickle.dump((salidas_esperadas, testing_set, result_matrix), f)
    print("Datos generados y guardados.")

# Asegurarse que los sets de parámetros son 2D para sklearn (N_samples, N_features)
if training_set.ndim == 1:
    training_set = training_set.reshape(-1, 1)
if testing_set.ndim == 1:
    testing_set = testing_set.reshape(-1, 1)
# Asegurarse que las salidas son 2D si tienen múltiples targets, o (N_samples,) si es uno solo
if salidas.ndim == 1:
    salidas = salidas.reshape(-1, 1)
if salidas_esperadas.ndim == 1:
    salidas_esperadas = salidas_esperadas.reshape(-1, 1)
# Si después de reshape tienen una sola columna y prefieres (N_samples,), puedes usar .ravel()
# Por ejemplo, si salidas.shape[1] == 1: salidas = salidas.ravel()
# Pero para RandomForestRegressor, (N_samples, 1) está bien.

# --- Función de evaluación generalizada (la misma que antes) ---
def evaluate_model_performance(
    model,
    X_train_transformed,
    y_train,
    X_test_transformed,
    y_test,
    reduced_basis,
    model_name="Modelo",
    params_train_original=None,
    params_test_original=None
):
    if params_train_original is None: params_train_original = X_train_transformed
    if params_test_original is None: params_test_original = X_test_transformed

    print(f"\n{BLUE}--- EVALUACIÓN: {model_name} ---{RESET}")

    y_pred_train = model.predict(X_train_transformed)
    y_pred_test = model.predict(X_test_transformed)

    # Asegurar que las predicciones tengan la misma forma que las y_train/y_test para np.matmul
    if y_pred_train.ndim == 1 and y_train.ndim == 2 and y_train.shape[1] > 0 :
        y_pred_train = y_pred_train.reshape(-1, y_train.shape[1])
    if y_pred_test.ndim == 1 and y_test.ndim == 2 and y_test.shape[1] > 0:
        y_pred_test = y_pred_test.reshape(-1, y_test.shape[1])


    print(f"\033[1;37;44m  Con error relativo  \033[0m")
    error_relativo_train = []
    for i in range(len(y_train)):
        sol_real_train = np.matmul(reduced_basis, y_train[i])
        sol_pred_train = np.matmul(reduced_basis, y_pred_train[i])
        norm_sol_real_train = np.linalg.norm(sol_real_train)
        if norm_sol_real_train < 1e-9:
             error_relativo_train.append(np.linalg.norm(sol_real_train - sol_pred_train))
        else:
            error_relativo_train.append(np.linalg.norm(sol_real_train - sol_pred_train) / norm_sol_real_train)
    
    print("ENTRENAMIENTO (Error Relativo)")
    print(f"Error Medio: {np.mean(error_relativo_train):.4e}")
    print(f"Varianza: {np.var(error_relativo_train):.4e}")
    print(f"Error Maximo: {np.max(error_relativo_train):.4e}")
    if len(error_relativo_train) > 0:
        arg_max_train = np.argmax(error_relativo_train)
        print(f"Parametro de error maximo: {params_train_original[arg_max_train]}")

    error_relativo_test = []
    for i in range(len(y_test)):
        sol_real_test = np.matmul(reduced_basis, y_test[i])
        sol_pred_test = np.matmul(reduced_basis, y_pred_test[i])
        norm_sol_real_test = np.linalg.norm(sol_real_test)
        if norm_sol_real_test < 1e-9:
            error_relativo_test.append(np.linalg.norm(sol_real_test - sol_pred_test))
        else:
            error_relativo_test.append(np.linalg.norm(sol_real_test - sol_pred_test) / norm_sol_real_test)

    print("TESTEO (Error Relativo)")
    print(f"Error Medio: {np.mean(error_relativo_test):.4e}")
    print(f"Varianza: {np.var(error_relativo_test):.4e}")
    print(f"Error Maximo: {np.max(error_relativo_test):.4e}")
    if len(error_relativo_test) > 0:
        arg_max_test = np.argmax(error_relativo_test)
        print(f"Parametro de error maximo: {params_test_original[arg_max_test]}")

    print(f"\n\033[1;37;44m  Con error máximo (componente a componente relativo al max de la solución real)  \033[0m")
    errores_max_comp_train = []
    for i in range(len(y_train)):
        sol_real_train = np.matmul(reduced_basis, y_train[i])
        sol_pred_train = np.matmul(reduced_basis, y_pred_train[i])
        res_abs_train = np.abs(sol_real_train - sol_pred_train)
        max_abs_sol_real_train = np.max(np.abs(sol_real_train))
        if max_abs_sol_real_train < 1e-9:
             errores_max_comp_train.append(np.max(res_abs_train))
        else:
            errores_max_comp_train.append(np.max(res_abs_train) / max_abs_sol_real_train)

    print("ENTRENAMIENTO (Error Máx Componente)")
    print(f"Error Medio: {np.mean(errores_max_comp_train):.4e}")
    print(f"Varianza: {np.var(errores_max_comp_train):.4e}")
    print(f"Error Maximo: {np.max(errores_max_comp_train):.4e}")
    if len(errores_max_comp_train) > 0:
        arg_max_comp_train = np.argmax(errores_max_comp_train)
        print(f"Parametro de error maximo: {params_train_original[arg_max_comp_train]}")

    errores_max_comp_test = []
    for i in range(len(y_test)):
        sol_real_test = np.matmul(reduced_basis, y_test[i])
        sol_pred_test = np.matmul(reduced_basis, y_pred_test[i])
        res_abs_test = np.abs(sol_real_test - sol_pred_test)
        max_abs_sol_real_test = np.max(np.abs(sol_real_test))
        if max_abs_sol_real_test < 1e-9:
            errores_max_comp_test.append(np.max(res_abs_test))
        else:
            errores_max_comp_test.append(np.max(res_abs_test) / max_abs_sol_real_test)
            
    print("TESTEO (Error Máx Componente)")
    print(f"Error Medio: {np.mean(errores_max_comp_test):.4e}")
    print(f"Varianza: {np.var(errores_max_comp_test):.4e}")
    print(f"Error Maximo: {np.max(errores_max_comp_test):.4e}")
    if len(errores_max_comp_test) > 0:
        arg_max_comp_test = np.argmax(errores_max_comp_test)
        print(f"Parametro de error maximo: {params_test_original[arg_max_comp_test]}")

# --- Escalado de Características de Entrada (X) ---
# RandomForest no es tan sensible al escalado como SVR o redes neuronales,
# pero escalar X no suele perjudicar y puede ayudar en algunos casos.
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(training_set)
X_test_scaled = scaler_X.transform(testing_set)

# (Opcional) Escalado de Características de Salida (Y)
# RandomForest es generalmente robusto a la escala de Y.
# No lo haremos por defecto aquí, pero puedes probarlo si los resultados no son buenos.
# scaler_Y = StandardScaler()
# Y_train_scaled = scaler_Y.fit_transform(salidas)
# Y_test_scaled_for_eval = scaler_Y.transform(salidas_esperadas) # Solo para evaluación si fuera necesario

# --- Ajuste de Hiperparámetros para RandomForestRegressor ---
# Define el modelo y la parrilla de parámetros
# Si 'salidas' es un vector (N_samples,), no necesitas MultiOutputRegressor.
# RandomForestRegressor maneja múltiples salidas directamente si 'salidas' es (N_samples, N_targets).
model_rf = RandomForestRegressor(random_state=42) # random_state para reproducibilidad

# Reduce la parrilla para una prueba más rápida, puedes expandirla después
param_grid_rf = {
    'n_estimators': [100, 200],        # Número de árboles. 100 es un buen inicio.
    'max_depth': [None, 10, 20],       # Profundidad máxima. None = nodos se expanden hasta pureza o min_samples_split.
    'min_samples_split': [2, 5, 10],   # Mínimo de muestras para dividir un nodo interno.
    'min_samples_leaf': [1, 2, 4],     # Mínimo de muestras en un nodo hoja.
    'max_features': ['sqrt', 'log2', 1.0] # Número de features a considerar para la mejor división.
                                      # 'auto' (ahora es 1.0 para RandomForestRegressor) o 'sqrt' o 'log2'
                                      # En versiones recientes de sklearn, 'auto' para RandomForestRegressor es equivalente a n_features (1.0)
                                      # y para RandomForestClassifier es 'sqrt'.
                                      # Usaremos 1.0 (todas las features), 'sqrt', 'log2'
}
# Si tienes pocas features (ej. 5), 'sqrt' (aprox 2) y 'log2' (aprox 2) pueden ser similares.
# 1.0 significa usar todas las features, lo que puede llevar a árboles más correlacionados.

print(f"\n{YELLOW}--- AJUSTANDO HIPERPARÁMETROS PARA RandomForestRegressor ---{RESET}")
# Usar una métrica de scoring apropiada, por ejemplo, neg_mean_squared_error
# O r2_score si quieres maximizar R^2
grid_search_rf = GridSearchCV(estimator=model_rf,
                              param_grid=param_grid_rf,
                              cv=3, # 3-fold cross-validation. Aumentar para más robustez si el tiempo lo permite.
                              scoring='neg_mean_squared_error', # Queremos minimizar MSE, así que maximizamos su negativo.
                              verbose=2, # Muestra progreso
                              n_jobs=-1) # Usa todos los procesadores disponibles

start_time_grid_rf = time.time()

# Si escalaste Y: grid_search_rf.fit(X_train_scaled, Y_train_scaled.ravel() si Y es (N,1) o Y_train_scaled)
# RandomForestRegressor puede tomar y como (n_samples,) o (n_samples, n_outputs)
# Si 'salidas' es (N,1), .ravel() lo convierte a (N,).
if salidas.shape[1] == 1:
    y_fit = salidas.ravel()
else:
    y_fit = salidas # Para múltiples salidas (N, n_outputs)

grid_search_rf.fit(X_train_scaled, y_fit)

duration_grid_rf = time.time() - start_time_grid_rf
print(f"{GREEN}Duración de GridSearchCV (RandomForest): {duration_grid_rf:.2f} segundos{RESET}")

print(f"{GREEN}Mejores hiperparámetros encontrados (RandomForest):{RESET}")
print(grid_search_rf.best_params_)

best_rf_model = grid_search_rf.best_estimator_

# --- Evaluación del Mejor Modelo RandomForest ---
# Si escalaste Y, recuerda hacer inverse_transform en las predicciones antes de evaluar.
# Por ahora, asumimos que no escalaste Y.

evaluate_model_performance(
    model=best_rf_model,
    X_train_transformed=X_train_scaled, # Usar datos escalados para predicción
    y_train=salidas,                   # Usar salidas originales para evaluación
    X_test_transformed=X_test_scaled,    # Usar datos escalados para predicción
    y_test=salidas_esperadas,          # Usar salidas esperadas originales para evaluación
    reduced_basis=reduced_basis,
    model_name="RandomForest (Mejores Hiperparámetros)",
    params_train_original=training_set, # Pasar los parámetros originales para el print
    params_test_original=testing_set
)

print(f"\n{YELLOW}--- PRUEBA DE RandomForestRegressor COMPLETADA ---{RESET}")