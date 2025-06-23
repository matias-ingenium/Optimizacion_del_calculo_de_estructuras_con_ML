import numpy as np
import time
import pickle
import os
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

GREEN = "\033[92m"
RESET = "\033[0m"

# --- Carga o generación de datos  ---
# Definir nombres de archivo para facilitar su modificación
training_file = "losa/elementos de prueba/prueba_train_4000_chico.pkl"
testing_file = "losa/elementos de prueba/prueba_test_30_chico.pkl"

if os.path.exists(training_file) and os.path.exists(testing_file):
    with open(training_file, "rb") as f:
        reduced_basis, salidas, training_set, snapshots_matrix = pickle.load(f)

    with open(testing_file, "rb") as f:
        salidas_esperadas, testing_set, result_matrix = pickle.load(f)

    print("Datos cargados desde archivo.")
else:
    from training_losa import training_data
    from testing_losa import testing_data
    start_time = time.time()
    
    reduced_basis, salidas, training_set, snapshots_matrix = training_data(4000)
    duration = time.time() - start_time
    print(f"{GREEN}Duración de training_data: {duration:.2f} segundos{RESET}")

    salidas_esperadas, testing_set, result_matrix = testing_data(30, reduced_basis)

    # Convertir a arrays de NumPy
    training_set = np.array(training_set)
    testing_set = np.array(testing_set)
    salidas = np.array(salidas)
    salidas_esperadas = np.array(salidas_esperadas)

    # Guardar archivos
    with open(training_file, "wb") as f:
        pickle.dump((reduced_basis, salidas, training_set, snapshots_matrix), f)

    with open(testing_file, "wb") as f:
        pickle.dump((salidas_esperadas, testing_set, result_matrix), f)

    print("Datos generados y guardados.")

# --- Entrenamiento con LGBMRegressor ---

# 1. Crear una instancia base de LGBMRegressor.

lgbm = LGBMRegressor(
    n_estimators=1000,       # Número de árboles a construir.
    learning_rate=0.1,      # Tasa de aprendizaje.
    num_leaves=60,           # Número máximo de hojas por árbol.
    random_state=77,         # Semilla para reproducibilidad.
    n_jobs=-1                # Usar todos los cores de la CPU.
)

# 2. Envolver el regresor base con MultiOutputRegressor para manejar múltiples salidas.
model = MultiOutputRegressor(estimator=lgbm)

print("Iniciando el entrenamiento con LGBMRegressor...")
start_time = time.time()

# 3. Entrenar el modelo con los datos de entrenamiento.
model.fit(training_set, salidas)

duration = time.time() - start_time
print(f"{GREEN}Duración de entrenamiento del modelo LGBM: {duration:.2f} segundos{RESET}")

# --- Evaluación del Modelo ---

# Predecir sobre todo el conjunto de datos de una sola vez (más eficiente)
predicciones_train = model.predict(training_set)
predicciones_test = model.predict(testing_set)

print(f"\n\033[1;37;44m  Con error relativo  \033[0m")

# Evaluación en ENTRENAMIENTO
error_relativo = []
for i in range(len(training_set)):
    norma_salida_real = np.linalg.norm(np.matmul(reduced_basis, salidas[i]))
    if norma_salida_real > 1e-4:
        res = np.matmul(reduced_basis, salidas[i]) - np.matmul(reduced_basis, predicciones_train[i])
        error = np.linalg.norm(res) / norma_salida_real
        error_relativo.append(error)

print("\nENTRENAMIENTO")
print("Error Medio: ", np.mean(error_relativo))
print("Varianza: ", np.var(error_relativo))
print("Error Maximo: ", np.max(error_relativo))
arg_max = np.argmax(error_relativo)
print("Parametro de error maximo: ", training_set[arg_max])

# Evaluación en TESTEO
error_relativo = []
for i in range(len(testing_set)):
    norma_salida_real = np.linalg.norm(np.matmul(reduced_basis, salidas_esperadas[i]))
    if norma_salida_real > 1e-4:
        res = np.matmul(reduced_basis, salidas_esperadas[i]) - np.matmul(reduced_basis, predicciones_test[i])
        error = np.linalg.norm(res) / norma_salida_real
        error_relativo.append(error)

print("\nTESTEO")
print("Error Medio: ", np.mean(error_relativo))
print("Varianza: ", np.var(error_relativo))
print("Error Maximo: ", np.max(error_relativo))
arg_max = np.argmax(error_relativo)
print("Parametro de error maximo: ", testing_set[arg_max])


print(f"\n\033[1;37;44m  Con error máximo  \033[0m")

# Evaluación en ENTRENAMIENTO
errores = []
for i in range(len(training_set)):
    salida_real_reconst = np.matmul(reduced_basis, salidas[i])
    max_abs_salida_real = np.max(np.abs(salida_real_reconst))
    if max_abs_salida_real > 1e-9:  # Evitar división por cero
        prediccion_reconst = np.matmul(reduced_basis, predicciones_train[i])
        res_abs = np.abs(salida_real_reconst - prediccion_reconst)
        errores.append(np.max(res_abs) / max_abs_salida_real)

print("\nENTRENAMIENTO")
print("Error Medio: ", np.mean(errores))
print("Varianza: ", np.var(errores))
print("Error Maximo: ", np.max(errores))
arg_max = np.argmax(errores)
print("Parametro de error maximo: ", training_set[arg_max])


# Evaluación en TESTEO
errores = []
for i in range(len(testing_set)):
    salida_real_reconst = np.matmul(reduced_basis, salidas_esperadas[i])
    max_abs_salida_real = np.max(np.abs(salida_real_reconst))
    if max_abs_salida_real > 1e-9: # Evitar división por cero
        prediccion_reconst = np.matmul(reduced_basis, predicciones_test[i])
        res_abs = np.abs(salida_real_reconst - prediccion_reconst)
        errores.append(np.max(res_abs) / max_abs_salida_real)

print("\nTESTEO")
print("Error Medio: ", np.mean(errores))
print("Varianza: ", np.var(errores))
print("Error Maximo: ", np.max(errores))
arg_max = np.argmax(errores)
print("Parametro de error maximo: ", testing_set[arg_max])