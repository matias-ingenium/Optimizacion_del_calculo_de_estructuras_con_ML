from redes import RED_NEURONAL
import torch
import numpy as np
import time
import pickle
import os
from multiprocessing import Pool, cpu_count

GREEN = "\033[92m"
RESET = "\033[0m"

import os

# --- Carga o generación de datos ---
data_dir = "losa/elementos de prueba" 
training_file = os.path.join(data_dir, "prueba_train_4000_17.pkl")
testing_file = os.path.join(data_dir, "prueba_test_30_17.pkl")

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
    salidas_esperadas, testing_set, result_matrix = testing_data(40, reduced_basis)
    training_set = np.array(training_set)
    testing_set = np.array(testing_set)
    salidas = np.array(salidas)
    salidas_esperadas = np.array(salidas_esperadas)
    with open(training_file, "wb") as f:
        pickle.dump((reduced_basis, salidas, training_set, snapshots_matrix), f)
    with open(testing_file, "wb") as f:
        pickle.dump((salidas_esperadas, testing_set, result_matrix), f)
    print("Datos generados y guardados.")

# --- Función de costo ---
def custom_loss_factory(reduced_basis_tensor):
    def loss(y_pred, y_true):
        recon_pred = torch.matmul(y_pred, reduced_basis_tensor.T)
        recon_true = torch.matmul(y_true, reduced_basis_tensor.T)
        return torch.mean((recon_pred - recon_true) ** 2) * 100
    return loss

reduced_basis_tensor = torch.tensor(reduced_basis, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
loss_func = custom_loss_factory(reduced_basis_tensor)

# --- ✨ FUNCIÓN PARA ENTRENAR Y EVALUAR UNA RED ✨ ---
def entrenar_y_evaluar_red(params):
    """
    Función que crea, entrena y evalúa una red neuronal con las métricas específicas.
    """
    id_red, config = params
    print(f"Iniciando entrenamiento para Red {id_red} con config: {config}")

    red = RED_NEURONAL(
        n_entrada=training_set.shape[1],
        m_salida=reduced_basis.shape[1],
        training_set=training_set,
        salidas=salidas,
        funcion_costo=loss_func,
        max_epochs=config["max_epochs"],
        capas_ocultas=config["capas_ocultas"],
        neuronas_por_capa=config["neuronas_por_capa"],
        lr=config["lr"],
        tol=0,
        min_delta=0,
        no_improvement_limit=3000
    )

    start_time = time.time()
    red.entrenar()
    duration = time.time() - start_time
    print(f"{GREEN}Red {id_red} entrenada en: {duration:.2f} segundos{RESET}")

    resultados_red = {
        "id": id_red, 
        "config": config, 
        "tiempo": duration,
        "train": {},
        "test": {}
    }

    # --- Evaluación en CONJUNTO DE ENTRENAMIENTO ---
    # Error relativo
    error_relativo_train = []
    for i in range(len(training_set)):
        if np.linalg.norm(np.matmul(reduced_basis, salidas[i])) > 1e-4:
            prediccion = red.predecir(training_set[i])
            res = np.matmul(reduced_basis, salidas[i]) - np.matmul(reduced_basis, prediccion[0])
            error_relativo_train.append(np.linalg.norm(res) / np.linalg.norm(np.matmul(reduced_basis, salidas[i])))
    
    # Error máximo
    errores_max_train = []
    for i in range(len(training_set)):
        if np.linalg.norm(np.matmul(reduced_basis, salidas[i])) > 1e-4:
            prediccion = red.predecir(training_set[i])
            res = abs(np.matmul(reduced_basis, salidas[i]) - np.matmul(reduced_basis, prediccion[0]))
            errores_max_train.append(np.max(res) / max(abs(np.matmul(reduced_basis, salidas[i]))))

    resultados_red["train"]["relativo"] = {"medio": np.mean(error_relativo_train), "var": np.var(error_relativo_train), "max": np.max(error_relativo_train)}
    resultados_red["train"]["maximo"] = {"medio": np.mean(errores_max_train), "var": np.var(errores_max_train), "max": np.max(errores_max_train)}

    # --- Evaluación en CONJUNTO DE TESTEO ---
    # Error relativo
    error_relativo_test = []
    for i in range(len(testing_set)):
        if np.linalg.norm(np.matmul(reduced_basis, salidas_esperadas[i])) > 1e-4:
            prediccion = red.predecir(testing_set[i])
            res = np.matmul(reduced_basis, salidas_esperadas[i]) - np.matmul(reduced_basis, prediccion[0])
            error_relativo_test.append(np.linalg.norm(res) / np.linalg.norm(np.matmul(reduced_basis, salidas_esperadas[i])))

    # Error máximo
    errores_max_test = []
    for i in range(len(testing_set)):
        if np.linalg.norm(np.matmul(reduced_basis, salidas_esperadas[i])) > 1e-4:
            prediccion = red.predecir(testing_set[i])
            res = abs(np.matmul(reduced_basis, salidas_esperadas[i]) - np.matmul(reduced_basis, prediccion[0]))
            errores_max_test.append(np.max(res) / max(abs(np.matmul(reduced_basis, salidas_esperadas[i]))))

    resultados_red["test"]["relativo"] = {"medio": np.mean(error_relativo_test), "var": np.var(error_relativo_test), "max": np.max(error_relativo_test)}
    resultados_red["test"]["maximo"] = {"medio": np.mean(errores_max_test), "var": np.var(errores_max_test), "max": np.max(errores_max_test)}
    
    return resultados_red

# --- 🚀 PARALELIZACIÓN DEL ENTRENAMIENTO 🚀 ---
import multiprocessing as mp

if __name__ == '__main__':
    # Establecer el método de inicio a 'spawn' antes de crear el Pool
    try:
        mp.set_start_method('spawn', force=True)
        print("Método de inicio de multiprocessing configurado como 'spawn'.")
    except RuntimeError:
        pass # Evita errores si ya está configurado
    
    # Define las distintas configuraciones de redes que quieres probar
    configuraciones = [
        {'capas_ocultas': 2, 'neuronas_por_capa': 150, 'lr': 0.008, 'max_epochs': 50000},
        {'capas_ocultas': 4, 'neuronas_por_capa': 150, 'lr': 0.008, 'max_epochs': 50000},
        {'capas_ocultas': 3, 'neuronas_por_capa': 100, 'lr': 0.008, 'max_epochs': 50000},
        {'capas_ocultas': 3, 'neuronas_por_capa': 150, 'lr': 0.01, 'max_epochs': 50000},
        {'capas_ocultas': 3, 'neuronas_por_capa': 200, 'lr': 0.008, 'max_epochs': 50000},
        {'capas_ocultas': 3, 'neuronas_por_capa': 100, 'lr': 0.008, 'max_epochs': 50000},
        {'capas_ocultas': 3, 'neuronas_por_capa': 250, 'lr': 0.008, 'max_epochs': 50000},
        {'capas_ocultas': 3, 'neuronas_por_capa': 300, 'lr': 0.008, 'max_epochs': 50000}
    ]

    parametros_entrenamiento = list(enumerate(configuraciones))

    num_procesos = min(cpu_count(), len(configuraciones))
    print(f"Usando {num_procesos} procesos para entrenar {len(configuraciones)} redes en paralelo.")

    start_time_total = time.time()
    # Usamos el Pool del módulo importado como mp
    with mp.Pool(processes=num_procesos) as pool:
        resultados = pool.map(entrenar_y_evaluar_red, parametros_entrenamiento)
    
    duration_total = time.time() - start_time_total
    print(f"\n{GREEN}Tiempo total de entrenamiento paralelo: {duration_total:.2f} segundos{RESET}")


    # --- 🏆 PRESENTACIÓN DE RESULTADOS DETALLADOS 🏆 ---
    resultados_ordenados = sorted(resultados, key=lambda x: x['test']['relativo']['medio'])
    
    print("\n" + "="*80)
    print("--- 🏆 Resultados Finales Ordenados por Error Relativo Medio en Testeo 🏆 ---")
    print("="*80)

    for res in resultados_ordenados:
        print(f"\n\033[1;37;44m --- Red ID: {res['id']} | Config: {res['config']} --- \033[0m")
        print(f"Tiempo de entrenamiento: {res['tiempo']:.2f}s")
        
        print("\n  \033[4mResultados en ENTRENAMIENTO:\033[0m")
        print("    Error Relativo -> Medio: {:.6f} | Varianza: {:.6f} | Máximo: {:.6f}".format(
            res['train']['relativo']['medio'], res['train']['relativo']['var'], res['train']['relativo']['max']))
        print("    Error Máximo   -> Medio: {:.6f} | Varianza: {:.6f} | Máximo: {:.6f}".format(
            res['train']['maximo']['medio'], res['train']['maximo']['var'], res['train']['maximo']['max']))

        print("\n  \033[4mResultados en TESTEO:\033[0m")
        print("    \033[92mError Relativo -> Medio: {:.6f}\033[0m | Varianza: {:.6f} | Máximo: {:.6f}".format(
            res['test']['relativo']['medio'], res['test']['relativo']['var'], res['test']['relativo']['max']))
        print("    Error Máximo   -> Medio: {:.6f} | Varianza: {:.6f} | Máximo: {:.6f}".format(
            res['test']['maximo']['medio'], res['test']['maximo']['var'], res['test']['maximo']['max']))
        print("-" * 80)