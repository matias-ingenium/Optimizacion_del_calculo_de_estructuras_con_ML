from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import pickle
import os

max_degree = 10
tolerance = 1e-2  # Error relativo aceptable
best_model = None
best_error = float("inf")

# Definir nombres de archivo para facilitar su modificación
training_file = "bloque_elastico/prueba_train.pkl"
testing_file = "bloque_elastico/prueba_test.pkl"

if os.path.exists(training_file) and os.path.exists(testing_file):
    with open(training_file, "rb") as f:
        reduced_basis, salidas, training_set, snapshots_matrix = pickle.load(f)

    with open(testing_file, "rb") as f:
        salidas_esperadas, testing_set, result_matrix = pickle.load(f)

    print("Datos cargados desde archivo.")
else:
    raise( "error")

X= training_set
Y= salidas

pred= True
start = time.time()
for degree in range(1, max_degree + 1):
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_poly, Y)
    
    # Predicción sobre entrenamiento
    Y_pred = model.predict(X_poly)
    
    # Reconstrucción completa para error relativo
    Y_recon = np.dot(Y, reduced_basis.T)
    Y_pred_recon = np.dot(Y_pred, reduced_basis.T)
    
    errores_relativos = np.linalg.norm(Y_recon - Y_pred_recon, axis=1) / np.linalg.norm(Y_recon, axis=1)
    error_medio = np.mean(errores_relativos)
    error_max = np.max(errores_relativos)
    
    print(f"Grado {degree} -> Error medio relativo: {error_medio:.6f}, Error máximo: {error_max}")
    
    if error_medio < best_error:
        best_error = error_medio
        best_model = (model, poly, degree)

    if error_medio < tolerance:
        print(f"\n✅ Se logró una buena aproximación con polinomio de grado {degree}")
        break
else:
    pred = False
    print(f"\n❌ No se logró una buena aproximación con polinomios hasta grado {max_degree}")

finish = time.time()
print("El entrenamiento es de: ", finish-start)
if pred:
    start = time.time()
    model, poly, grado = best_model
    X_test_poly = poly.transform(testing_set)
    Y_test_pred = model.predict(X_test_poly)

    Y_test_recon = np.dot(salidas_esperadas, reduced_basis.T)
    Y_test_pred_recon = np.dot(Y_test_pred, reduced_basis.T)
    finish = time.time()
    errores_test = np.linalg.norm(Y_test_recon - Y_test_pred_recon, axis=1) / np.linalg.norm(Y_test_recon, axis=1)
    print("TESTING")
    print("Error Medio: ", np.mean(errores_test))
    print("Error Máximo: ", np.max(errores_test))
    print("Tiempo online medio: ", (finish-start)/len(testing_set))