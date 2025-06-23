import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# Definir nombres de archivo para facilitar su modificación
training_file = "losa/elementos de prueba/prueba_train_4000_chico.pkl"
testing_file = "losa/elementos de prueba/prueba_test_40_chico.pkl"


if os.path.exists(training_file) and os.path.exists(testing_file):
    with open(training_file, "rb") as f:
        reduced_basis, salidas, training_set, snapshots_matrix = pickle.load(f)

    with open(testing_file, "rb") as f:
        salidas_esperadas, testing_set, result_matrix = pickle.load(f)

    print("Datos cargados desde archivo.")


# Parámetros
i = 0 # columna de training_set
j = 3  # columna de salidas
max_degree = 10

# Extraer columnas
x = training_set[:, i].reshape(-1, 1)
y = salidas[:, j]

corr = np.corrcoef(training_set[:, i], salidas[:, j])[0, 1]
print(f"Correlación entre training_set[:, {i}] y salidas[:, {j}]: {corr:.4f}")

best_r2 = -np.inf
best_model = None
best_degree = None
best_poly = None
best_y_pred = None

# Probar distintos grados de polinomio
for degree in range(1, max_degree + 1):
    poly = PolynomialFeatures(degree)
    x_poly = poly.fit_transform(x)
    model = LinearRegression().fit(x_poly, y)
    y_pred = model.predict(x_poly)
    r2 = r2_score(y, y_pred)
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_degree = degree
        best_poly = poly
        best_y_pred = y_pred

# Resultados
print(f"Mejor grado: {best_degree}")
print(f"R²: {best_r2:.4f}")
print(f"Coeficientes: {best_model.coef_}")
print(f"Intercepto: {best_model.intercept_}")

# Graficar
x_plot = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
x_plot_poly = best_poly.transform(x_plot)
y_plot = best_model.predict(x_plot_poly)

plt.scatter(x, y, alpha=0.6, label='Datos')
plt.plot(x_plot, y_plot, color='red', label=f'Polinomio grado {best_degree}')
plt.xlabel(f'training_set[:, {i}]')
plt.ylabel(f'salidas[:, {j}]')
plt.title(f'Mejor ajuste polinomial (grado {best_degree}, R² = {best_r2:.4f})')
plt.legend()
plt.grid(True)
plt.show()
