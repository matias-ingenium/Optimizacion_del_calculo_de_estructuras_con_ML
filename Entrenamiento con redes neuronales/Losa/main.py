
from redes import RED_NEURONAL
import torch
import numpy as np
import time
import pickle
import os

GREEN = "\033[92m"
RESET = "\033[0m"

# Definir nombres de archivo para facilitar su modificación
training_file = "prueba_train_4000_chico.pkl"
testing_file = "prueba_test_40_chico.pkl"

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
    
    
    reduced_basis, salidas, training_set, snapshots_matrix = training_data(
        4000
    )
    duration = time.time() - start_time

    print(f"{GREEN}Duración de training_data: {duration:.2f} segundos{RESET}")

    salidas_esperadas, testing_set, result_matrix = testing_data(
        40, reduced_basis
    )

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




def custom_loss_factory(reduced_basis_tensor):
    def loss(y_pred, y_true):
        recon_pred = torch.matmul(y_pred, reduced_basis_tensor.T)
        recon_true = torch.matmul(y_true, reduced_basis_tensor.T)
        return torch.mean((recon_pred - recon_true) ** 2) * 100
    return loss


# Crear el tensor de base reducida
reduced_basis_tensor = torch.tensor(reduced_basis, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")

# Crear la función de pérdida personalizada ya con la base reducida incluida
loss_func = custom_loss_factory(reduced_basis_tensor)

red2 = RED_NEURONAL(
    n_entrada=training_set.shape[1],
    m_salida= reduced_basis.shape[1],
    training_set=training_set,
    salidas=salidas,
    funcion_costo= loss_func,
    max_epochs=50000,
    capas_ocultas=2,
    neuronas_por_capa=140,
    lr=0.008,
    tol=0,
    min_delta=0,
    no_improvement_limit=3000

)

start_time = time.time()
red2.entrenar()
duration = time.time() - start_time
print(f"{GREEN}Duración de entrenamiento de la red: {duration:.2f} segundos{RESET}")

print(f"\033[1;37;44m  Con error relativo  \033[0m")
error_relativo=[]
for i in range(len(training_set)):

  if np.linalg.norm(np.matmul(reduced_basis,salidas[i]))>1e-4:
    prediccion = red2.predecir(training_set[i])

    res=np.matmul(reduced_basis,salidas[i])-np.matmul(reduced_basis,prediccion[0])
    error_relativo.append(np.linalg.norm(res)/np.linalg.norm(np.matmul(reduced_basis,salidas[i])))

print("ENTRENAMIENTO")
print("Error Medio: ", np.mean(error_relativo))
print("Varianza: ", np.var(error_relativo))
print("Error Maximo: ", np.max(error_relativo))
arg_max = np.argmax(error_relativo)
print("Parametro de error maximo: ", training_set[arg_max])
print("Norma del error máximo: ", np.linalg.norm(np.matmul(reduced_basis,salidas[arg_max])))


error_relativo=[]
for i in range(len(testing_set)):
  
  if np.linalg.norm(np.matmul(reduced_basis,salidas_esperadas[i]))>1e-4:
    prediccion = red2.predecir(testing_set[i])

    res=np.matmul(reduced_basis,salidas_esperadas[i])-np.matmul(reduced_basis,prediccion[0])
    error_relativo.append(np.linalg.norm(res)/np.linalg.norm(np.matmul(reduced_basis,salidas_esperadas[i])))

print("TESTEO")
print("Error Medio: ", np.mean(error_relativo))
print("Varianza: ", np.var(error_relativo))
print("Error Maximo: ", np.max(error_relativo))
arg_max = np.argmax(error_relativo)
print("Parametro de error maximo: ", testing_set[arg_max])
print("Norma del error máximo: ", np.linalg.norm(np.matmul(reduced_basis,salidas_esperadas[arg_max])))

print(f"\033[1;37;44m  Con error máximo  \033[0m")

errores=[]
for i in range(len(training_set)):
    if np.linalg.norm(np.matmul(reduced_basis,salidas[i]))>1e-4:
        prediccion = red2.predecir(training_set[i])

        res = abs(np.matmul(reduced_basis,salidas[i])-np.matmul(reduced_basis,prediccion[0]))
        errores.append(np.max(res)/max(abs(np.matmul(reduced_basis,salidas[i]))))

print("ENTRENAMIENTO")
print("Error Medio: ", np.mean(errores))
print("Varianza: ", np.var(errores))
print("Error Maximo: ", np.max(errores))
arg_max = np.argmax(errores)
print("Parametro de error maximo: ", training_set[arg_max])
print("Norma del error máximo: ", np.linalg.norm(np.matmul(reduced_basis,salidas[arg_max])))


errores=[]
for i in range(len(testing_set)):
    if np.linalg.norm(np.matmul(reduced_basis,salidas_esperadas[i]))>1e-4:
        prediccion = red2.predecir(testing_set[i])

        res = abs(np.matmul(reduced_basis,salidas_esperadas[i])-np.matmul(reduced_basis,prediccion[0]))
        errores.append(np.max(res)/max(abs(np.matmul(reduced_basis,salidas_esperadas[i]))))

print("TESTEO")
print("Error Medio: ", np.mean(errores))
print("Varianza: ", np.var(errores))
print("Error Maximo: ", np.max(errores))
arg_max = np.argmax(errores)
print("Parametro de error maximo: ", testing_set[arg_max])
print("Norma del error máximo: ", np.linalg.norm(np.matmul(reduced_basis,salidas_esperadas[arg_max])))

