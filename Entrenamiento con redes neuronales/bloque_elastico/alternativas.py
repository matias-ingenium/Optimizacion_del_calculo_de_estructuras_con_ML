from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
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



rbf_feature = RBFSampler(gamma=1.0, n_components=5000)
model = make_pipeline(rbf_feature, Ridge(alpha=1e-3))
model.fit(X, Y)

# Predicción sobre entrenamiento
Y_pred = model.predict(X)

# Reconstrucción completa para error relativo
Y_recon = np.dot(Y, reduced_basis.T)
Y_pred_recon = np.dot(Y_pred, reduced_basis.T)

errores_relativos = np.linalg.norm(Y_recon - Y_pred_recon, axis=1) / np.linalg.norm(Y_recon, axis=1)
error_medio = np.mean(errores_relativos)

print(error_medio)