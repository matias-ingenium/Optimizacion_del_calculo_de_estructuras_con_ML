from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import os
import pickle
import time
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# --- Constantes para colores ---
GREEN = "\033[92m"
RESET = "\033[0m"
BLUE = "\033[94m"
YELLOW = "\033[93m"


# Definir nombres de archivo para facilitar su modificación
training_file = "losa/elementos de prueba/prueba_train_4000_chico.pkl"
testing_file = "losa/elementos de prueba/prueba_test_40_chico.pkl"

# --- Carga o generación de datos ---
if os.path.exists(training_file) and os.path.exists(testing_file):
    with open(training_file, "rb") as f:
        reduced_basis, salidas, training_set, snapshots_matrix = pickle.load(f)
    with open(testing_file, "rb") as f:
        salidas_esperadas, testing_set, result_matrix = pickle.load(f)
    print("Datos cargados desde archivo.")
else:
    # Asumimos que tienes estos módulos disponibles y funcionan
    from training_losa import training_data
    from testing_losa import testing_data
    print("Generando datos...")
    start_time = time.time()
    reduced_basis, salidas, training_set, snapshots_matrix = training_data(1000)
    duration = time.time() - start_time
    print(f"{GREEN}Duración de training_data: {duration:.2f} segundos{RESET}")

    salidas_esperadas, testing_set, result_matrix = testing_data(30, reduced_basis)

    training_set = np.array(training_set)
    testing_set = np.array(testing_set)
    salidas = np.array(salidas)
    salidas_esperadas = np.array(salidas_esperadas)

    with open(training_file, "wb") as f:
        pickle.dump((reduced_basis, salidas, training_set, snapshots_matrix), f)
    with open(testing_file, "wb") as f:
        pickle.dump((salidas_esperadas, testing_set, result_matrix), f)
    print("Datos generados y guardados.")

# Escalar características de entrada
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(training_set)
X_test_scaled = scaler_X.transform(testing_set)



# Define el modelo base y el MultiOutputRegressor si es necesario
if salidas.ndim > 1 and salidas.shape[1] > 1:
    base_estimator = SVR(kernel='rbf')
    model_to_tune = MultiOutputRegressor(base_estimator)
    # Los parámetros para GridSearchCV deben prefijarse con 'estimator__'
    param_grid = {
        'estimator__C': [0.1, 1, 10, 100],
        'estimator__gamma': [0.001, 0.01, 0.1, 1, 'scale'],
        'estimator__epsilon': [0.01, 0.1, 0.5]
    }
else:
    model_to_tune = SVR(kernel='rbf')
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1, 'scale'],
        'epsilon': [0.01, 0.1, 0.5]
    }

print(f"\n{YELLOW}--- AJUSTANDO HIPERPARÁMETROS PARA SVR ---{RESET}")
# Usar una métrica de scoring apropiada, por ejemplo, neg_mean_squared_error
# GridSearchCV maximiza el score, por eso se usa el negativo del MSE.
grid_search = GridSearchCV(model_to_tune, param_grid, cv=3, # 3-fold cross-validation
                           scoring='neg_mean_squared_error',
                           verbose=2, n_jobs=-1) # n_jobs=-1 usa todos los procesadores

start_time_grid = time.time()
# Si escalaste Y: grid_search.fit(X_train_scaled, Y_train_scaled)
grid_search.fit(X_train_scaled, salidas)
duration_grid = time.time() - start_time_grid
print(f"{GREEN}Duración de GridSearchCV: {duration_grid:.2f} segundos{RESET}")

print("Mejores hiperparámetros encontrados:")
print(grid_search.best_params_)

best_svr_model = grid_search.best_estimator_


def evaluate_model_performance(
    model,
    X_train_transformed, # Características ya transformadas
    y_train,
    X_test_transformed,  # Características ya transformadas
    y_test,
    reduced_basis,
    model_name="Modelo Lineal",
    params_train_original=None, # Para imprimir el parámetro original
    params_test_original=None   # Para imprimir el parámetro original
):
    """
    Evalúa el rendimiento de un modelo entrenado.
    model: modelo de sklearn entrenado.
    X_train_transformed, y_train: datos de entrenamiento (features transformadas, salidas).
    X_test_transformed, y_test: datos de testeo (features transformadas, salidas esperadas).
    reduced_basis: la base reducida para reconstruir la solución.
    model_name: nombre del modelo para los prints.
    params_train_original, params_test_original: parámetros originales antes de la transformación.
    """
    if params_train_original is None: params_train_original = X_train_transformed
    if params_test_original is None: params_test_original = X_test_transformed

    print(f"\n{BLUE}--- EVALUACIÓN: {model_name} ---{RESET}")

    # Predicciones
    # El modelo ya está entrenado, solo predecimos
    y_pred_train = model.predict(X_train_transformed)
    y_pred_test = model.predict(X_test_transformed)

    # --- Evaluación con Error Relativo ---
    print(f"\033[1;37;44m  Con error relativo  \033[0m")
    
    # Entrenamiento
    error_relativo_train = []
    for i in range(len(y_train)):
        sol_real_train = np.matmul(reduced_basis, y_train[i])
        sol_pred_train = np.matmul(reduced_basis, y_pred_train[i])
        norm_sol_real_train = np.linalg.norm(sol_real_train)
        if norm_sol_real_train < 1e-6: # Evitar división por cero si la norma es muy pequeña
             error_relativo_train.append(np.linalg.norm(sol_real_train - sol_pred_train)) # Error absoluto en este caso
        else:
            error_relativo_train.append(np.linalg.norm(sol_real_train - sol_pred_train) / norm_sol_real_train)
    
    print("ENTRENAMIENTO (Error Relativo)")
    print(f"Error Medio: {np.mean(error_relativo_train):.4e}")
    print(f"Varianza: {np.var(error_relativo_train):.4e}")
    print(f"Error Maximo: {np.max(error_relativo_train):.4e}")
    arg_max_train = np.argmax(error_relativo_train)
    print(f"Parametro de error maximo: {params_train_original[arg_max_train]}")
 


    # Testeo
    error_relativo_test = []
    for i in range(len(y_test)):
        sol_real_test = np.matmul(reduced_basis, y_test[i])
        sol_pred_test = np.matmul(reduced_basis, y_pred_test[i])
        norm_sol_real_test = np.linalg.norm(sol_real_test)
        if norm_sol_real_test < 1e-6:
            error_relativo_test.append(np.linalg.norm(sol_real_test - sol_pred_test))
        else:
            error_relativo_test.append(np.linalg.norm(sol_real_test - sol_pred_test) / norm_sol_real_test)

    print("TESTEO (Error Relativo)")
    print(f"Error Medio: {np.mean(error_relativo_test):.4e}")
    print(f"Varianza: {np.var(error_relativo_test):.4e}")
    print(f"Error Maximo: {np.max(error_relativo_test):.4e}")
    arg_max_test = np.argmax(error_relativo_test)
    print(f"Parametro de error maximo: {params_test_original[arg_max_test]}")


    # --- Evaluación con Error Máximo (componente a componente) ---
    print(f"\n\033[1;37;44m  Con error máximo (componente a componente relativo al max de la solución real)  \033[0m")

    # Entrenamiento
    errores_max_comp_train = []
    for i in range(len(y_train)):
        sol_real_train = np.matmul(reduced_basis, y_train[i])
        sol_pred_train = np.matmul(reduced_basis, y_pred_train[i])
        res_abs_train = np.abs(sol_real_train - sol_pred_train)
        max_abs_sol_real_train = np.max(np.abs(sol_real_train))
        if max_abs_sol_real_train < 1e-6:
             errores_max_comp_train.append(np.max(res_abs_train)) # Error absoluto en este caso
        else:
            errores_max_comp_train.append(np.max(res_abs_train) / max_abs_sol_real_train)

    print("ENTRENAMIENTO (Error Máx Componente)")
    print(f"Error Medio: {np.mean(errores_max_comp_train):.4e}")
    print(f"Varianza: {np.var(errores_max_comp_train):.4e}")
    print(f"Error Maximo: {np.max(errores_max_comp_train):.4e}")
    arg_max_comp_train = np.argmax(errores_max_comp_train)
    print(f"Parametro de error maximo: {params_train_original[arg_max_comp_train]}")


    # Testeo
    errores_max_comp_test = []
    for i in range(len(y_test)):
        sol_real_test = np.matmul(reduced_basis, y_test[i])
        sol_pred_test = np.matmul(reduced_basis, y_pred_test[i])
        res_abs_test = np.abs(sol_real_test - sol_pred_test)
        max_abs_sol_real_test = np.max(np.abs(sol_real_test))
        if max_abs_sol_real_test < 1e-6:
            errores_max_comp_test.append(np.max(res_abs_test))
        else:
            errores_max_comp_test.append(np.max(res_abs_test) / max_abs_sol_real_test)
            
    print("TESTEO (Error Máx Componente)")
    print(f"Error Medio: {np.mean(errores_max_comp_test):.4e}")
    print(f"Varianza: {np.var(errores_max_comp_test):.4e}")
    print(f"Error Maximo: {np.max(errores_max_comp_test):.4e}")
    arg_max_comp_test = np.argmax(errores_max_comp_test)
    print(f"Parametro de error maximo: {params_test_original[arg_max_comp_test]}")


evaluate_model_performance(
    model=best_svr_model,
    X_train_transformed=X_train_scaled,
    y_train=salidas,
    X_test_transformed=X_test_scaled,
    y_test=salidas_esperadas,
    reduced_basis=reduced_basis,
    model_name="SVR (Mejores Hiperparámetros)",
    params_train_original=training_set,
    params_test_original=testing_set
)