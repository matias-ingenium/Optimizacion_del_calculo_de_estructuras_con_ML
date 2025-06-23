import os
import pickle
import time
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge # Ridge es una buena alternativa con regularización
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# --- Constantes para colores (opcional, para mejorar la legibilidad del output) ---
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

# Asegurarse que los sets de parámetros son 2D para sklearn (N_samples, N_features)
if training_set.ndim == 1:
    training_set = training_set.reshape(-1, 1)
if testing_set.ndim == 1:
    testing_set = testing_set.reshape(-1, 1)

# --- Función para crear características personalizadas ---
def create_custom_features(X, poly_degree=2, include_inverse_powers=True, epsilon=1e-8):
    """
    Genera características polinómicas y/o potencias inversas.
    X: array de entrada (N_samples, N_features_input)
    poly_degree: grado del polinomio
    include_inverse_powers: booleano para incluir x^-1, x^-2, x^-3
    epsilon: pequeño valor para evitar división por cero
    """
    features_list = []

    # 1. Características Polinómicas
    if poly_degree > 0:
        poly = PolynomialFeatures(degree=poly_degree, include_bias=False) # Bias lo maneja LinearRegression
        poly_features = poly.fit_transform(X)
        features_list.append(poly_features)

    # 2. Características de Potencias Inversas (aplicadas a cada característica original)
    if include_inverse_powers:
        for i in range(X.shape[1]): # Para cada característica original en X
            X_col = X[:, i]
            inv1 = 1.0 / (X_col + np.sign(X_col) * epsilon + (X_col == 0) * epsilon) # Evita div por cero
            inv2 = 1.0 / (X_col**2 + epsilon) # Asumiendo X_col**2 siempre >=0
            inv3 = 1.0 / (X_col**3 + np.sign(X_col**3) * epsilon + (X_col**3 == 0) * epsilon)
            
            # Si X_col puede ser cero, hay que tener más cuidado con inv2 y inv3 si X_col es negativo
            # Para X_col**2, X_col**2 + epsilon es seguro si epsilon > 0.
            # Para X_col**3, si X_col es negativo, X_col**3 es negativo.
            # Una forma más robusta para potencias negativas:
            # inv1 = np.power(X_col + epsilon * np.sign(X_col) + epsilon * (X_col==0), -1)
            # inv2 = np.power(X_col + epsilon * np.sign(X_col) + epsilon * (X_col==0), -2) # Esto daría (X+eps)^-2
            # O aplicar a X^2, X^3 directamente:
            # inv2 = 1.0 / (np.power(X_col, 2) + epsilon)
            # inv3 = 1.0 / (np.power(X_col, 3) + np.sign(np.power(X_col,3))*epsilon + (np.power(X_col,3)==0)*epsilon)


            features_list.append(inv1.reshape(-1, 1))
            features_list.append(inv2.reshape(-1, 1))
            features_list.append(inv3.reshape(-1, 1))
            
    if not features_list: # Si no se seleccionó ninguna, usar originales
        return X

    return np.hstack(features_list)

# --- Función de evaluación generalizada ---
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
        if norm_sol_real_train < 1e-9: # Evitar división por cero si la norma es muy pequeña
             error_relativo_train.append(np.linalg.norm(sol_real_train - sol_pred_train)) # Error absoluto en este caso
        else:
            error_relativo_train.append(np.linalg.norm(sol_real_train - sol_pred_train) / norm_sol_real_train)
    
    print("ENTRENAMIENTO (Error Relativo)")
    print(f"Error Medio: {np.mean(error_relativo_train):.4e}")
    print(f"Varianza: {np.var(error_relativo_train):.4e}")
    print(f"Error Maximo: {np.max(error_relativo_train):.4e}")
    arg_max_train = np.argmax(error_relativo_train)
    print(f"Parametro de error maximo: {params_train_original[arg_max_train]}")
    # print(f"Norma de la solución real (error máximo): {np.linalg.norm(np.matmul(reduced_basis, y_train[arg_max_train])):.4e}")


    # Testeo
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
    arg_max_test = np.argmax(error_relativo_test)
    print(f"Parametro de error maximo: {params_test_original[arg_max_test]}")
    # print(f"Norma de la solución real (error máximo): {np.linalg.norm(np.matmul(reduced_basis, y_test[arg_max_test])):.4e}")

    # --- Evaluación con Error Máximo (componente a componente) ---
    print(f"\n\033[1;37;44m  Con error máximo (componente a componente relativo al max de la solución real)  \033[0m")

    # Entrenamiento
    errores_max_comp_train = []
    for i in range(len(y_train)):
        sol_real_train = np.matmul(reduced_basis, y_train[i])
        sol_pred_train = np.matmul(reduced_basis, y_pred_train[i])
        res_abs_train = np.abs(sol_real_train - sol_pred_train)
        max_abs_sol_real_train = np.max(np.abs(sol_real_train))
        if max_abs_sol_real_train < 1e-9:
             errores_max_comp_train.append(np.max(res_abs_train)) # Error absoluto en este caso
        else:
            errores_max_comp_train.append(np.max(res_abs_train) / max_abs_sol_real_train)

    print("ENTRENAMIENTO (Error Máx Componente)")
    print(f"Error Medio: {np.mean(errores_max_comp_train):.4e}")
    print(f"Varianza: {np.var(errores_max_comp_train):.4e}")
    print(f"Error Maximo: {np.max(errores_max_comp_train):.4e}")
    arg_max_comp_train = np.argmax(errores_max_comp_train)
    print(f"Parametro de error maximo: {params_train_original[arg_max_comp_train]}")
    # print(f"Norma de la solución real (error máximo): {np.linalg.norm(np.matmul(reduced_basis, y_train[arg_max_comp_train])):.4e}")

    # Testeo
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
    arg_max_comp_test = np.argmax(errores_max_comp_test)
    print(f"Parametro de error maximo: {params_test_original[arg_max_comp_test]}")
    # print(f"Norma de la solución real (error máximo): {np.linalg.norm(np.matmul(reduced_basis, y_test[arg_max_comp_test])):.4e}")


# --- Bucle para probar diferentes configuraciones de modelos ---
print(f"\n{YELLOW}--- INICIANDO PRUEBAS DE MODELOS LINEALES ---{RESET}")

# Configuraciones a probar: (grado_polinomio, usar_inversas)
configurations = [
    (0, False), # Solo parámetros originales (si create_custom_features devuelve X) - Necesita ajuste en create_custom_features o no usarla.
                # Para este caso, usaremos directamente training_set y testing_set sin transformar si poly_degree=0 y no inversas.
    (1, False), # Linealidad simple
    (2, False), # Polinomio grado 2
    (3, False), # Polinomio grado 3
    (1, True),  # Lineal + Inversas
    (2, True),  # Polinomio grado 2 + Inversas
    (3, True),  # Polinomio grado 3 + Inversas
]

for poly_deg, use_inv in configurations:
    model_name = f"PolyDeg={poly_deg}, InvPow={use_inv}"
    print(f"\n{GREEN}Probando configuración: {model_name}{RESET}")

    # 1. Crear características
    if poly_deg == 0 and not use_inv:
        # Caso especial: usar características originales sin transformación polinómica ni inversa
        # Asumimos que create_custom_features con poly_degree=0 y include_inverse_powers=False
        # debería devolver X tal cual, o simplemente no la llamamos para este caso.
        # Por simplicidad, si no hay transformaciones, X_train_tf = training_set
        # Sin embargo, create_custom_features tal como está, si no hay features_list, devuelve X.
        # O podrías tener X_train_tf = training_set.copy()
        X_train_transformed = create_custom_features(training_set, poly_degree=0, include_inverse_powers=False) # Devuelve X original
        X_test_transformed = create_custom_features(testing_set, poly_degree=0, include_inverse_powers=False)   # Devuelve X original
    else:
        X_train_transformed = create_custom_features(training_set, poly_degree=poly_deg, include_inverse_powers=use_inv)
        X_test_transformed = create_custom_features(testing_set, poly_degree=poly_deg, include_inverse_powers=use_inv)
    
    print(f"Dimensiones de características transformadas (entrenamiento): {X_train_transformed.shape}")

    # (Opcional pero recomendado) Escalar características después de la transformación
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_transformed)
    X_test_scaled = scaler.transform(X_test_transformed)
    
    # 2. Definir y entrenar el modelo
    # Usaremos LinearRegression. Podrías probar Ridge(alpha=0.1) también.
    linear_model = LinearRegression(fit_intercept=True)
    # linear_model = Ridge(alpha=1.0) # Prueba con Ridge también

    start_time_train = time.time()
    linear_model.fit(X_train_scaled, salidas) # Entrenar con 'salidas'
    duration_train = time.time() - start_time_train
    print(f"{GREEN}Duración de entrenamiento de {model_name}: {duration_train:.2f} segundos{RESET}")

    # 3. Evaluar el modelo
    evaluate_model_performance(
        model=linear_model,
        X_train_transformed=X_train_scaled, # Usar datos escalados para predicción
        y_train=salidas,
        X_test_transformed=X_test_scaled,   # Usar datos escalados para predicción
        y_test=salidas_esperadas,
        reduced_basis=reduced_basis,
        model_name=model_name,
        params_train_original=training_set, # Pasar los parámetros originales para el print
        params_test_original=testing_set
    )

print(f"\n{YELLOW}--- PRUEBAS DE MODELOS LINEALES COMPLETADAS ---{RESET}")