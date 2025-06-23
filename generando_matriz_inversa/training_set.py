# training_set.py (CORRECCIÓN PARA EVITAR 0 MODOS)
# 12 parámetros
from dolfin import *
import numpy as np
from multiprocessing import Pool
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
from cant_modos import *
import pickle
import os

def convert_dolfin_to_scipy_sparse(dolfin_matrix):
    backend_matrix = as_backend_type(dolfin_matrix)
    indptr, indices, data = backend_matrix.mat().getValuesCSR()
    dim1 = backend_matrix.size(0)
    dim2 = backend_matrix.size(1)
    return csr_matrix((data, indices, indptr), shape=(dim1, dim2))



def solve_snapshot(args):
    mu, mesh_file, facet_file, physical_file, lambda_1, lambda_2 = args
    mesh = Mesh(mesh_file)
    subdomains = MeshFunction("size_t", mesh, physical_file)
    boundaries = MeshFunction("size_t", mesh, facet_file)
    V = VectorFunctionSpace(mesh, "P", 1)
    bc = DirichletBC(V, Constant((0.0, 0.0)), lambda x, on_boundary: on_boundary and near(x[0], 0.0))
    dx = Measure("dx")(subdomain_data=subdomains)
    ds = Measure("ds")(subdomain_data=boundaries)
    u = TrialFunction(V)
    v = TestFunction(V)

    a =  (mu[0]*19/2+21/2) *(2.0 * lambda_2 * inner(sym(grad(u)), sym(grad(v))) + 
                lambda_1 * tr(sym(grad(u))) * tr(sym(grad(v)))) * dx(1) 

    for i in range(1, 9):
        a += (mu[i]*19/2+21/2)  * (2.0 * lambda_2 * inner(sym(grad(u)), sym(grad(v))) + #Ahora con esto la entrada puede ser entre (0,1)
                      lambda_1 * tr(sym(grad(u))) * tr(sym(grad(v)))) * dx(i+1)
    f_load = Constant((1.0, 0.0))

    L = mu[1+8] * inner(f_load, v) * ds(2) + mu[2+8] * inner(f_load, v) * ds(3) + mu[3+8] * inner(f_load, v) * ds(4)
    A_dolfin, fv = assemble_system(a, L, [bc])
    sol = Function(V)
    solve(A_dolfin, sol.vector(), fv)
    A_scipy_sparse = convert_dolfin_to_scipy_sparse(A_dolfin)
    return sol.vector().get_local(), A_scipy_sparse, fv.get_local()

def training_data(n_snapshots, mesh_file, facet_file, physical_file):
    """
    Genera los datos de entrenamiento. Carga los parámetros y resultados
    si ya existen, o los calcula y guarda si no.
    """
    print(f"INICIO DE GENERACIÓN DE TRAINING DATA: {n_snapshots} snapshots")
    
    # 1. Definir el nombre del archivo de guardado
    snapshot_data_file = f"generando_matriz_inversa/elementos de prueba/datos_entrenamiento_{n_snapshots}.pkl"

    # 2. Comprobar si el archivo de datos ya existe
    if os.path.exists(snapshot_data_file):
        print(f"Cargando datos pre-calculados desde '{snapshot_data_file}'...")
        with open(snapshot_data_file, "rb") as f:
            # Cargar y desempaquetar la tupla que contiene ambos: parámetros y resultados
            training_set, results = pickle.load(f)
        print("Carga completada.")
    else:
        print(f"No se encontraron datos guardados. Generando parámetros y ejecutando la simulación...")
        
        # --- Bloque de cálculo pesado ---
        
        # Generar los parámetros de entrada ANTES de la simulación
        training_set = [tuple(random.uniform(-1, 1) for _ in range(12)) for _ in range(n_snapshots)]
        
        # Configuración de FEniCS
        E, nu = 1.0, 0.3
        lambda_1 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        lambda_2 = E / (2.0 * (1.0 + nu))
        args_for_snapshots = [(mu, mesh_file, facet_file, physical_file, lambda_1, lambda_2) for mu in training_set]
        
        # Ejecutar el cálculo pesado en paralelo
        with Pool() as pool:
            results = list(tqdm(pool.imap(solve_snapshot, args_for_snapshots), total=len(training_set), desc="Procesando snapshots"))

        # Guardar los resultados y los parámetros JUNTOS
        print(f"Guardando parámetros y resultados en '{snapshot_data_file}'...")
        with open(snapshot_data_file, "wb") as f:
            # Guardamos una tupla que contiene ambas variables
            pickle.dump((training_set, results), f)
        print("Guardado completado.")
        # --- Fin del bloque de cálculo pesado ---

    # 3. Procesar los resultados (este código se ejecuta siempre)
    # 'results' y 'training_set' están ahora correctamente sincronizados
    solutions, list_A_scipy, list_fv = zip(*results)
    snapshots_matrix = np.array(solutions).T
    
    print("SNAPSHOTS OBTENIDOS Y PROCESADOS")

    V_svd, sigma, Wt = np.linalg.svd(snapshots_matrix, full_matrices=False)
    print(f"Valores singulares de desplazamientos: {sigma}")
    n_modos_u = contar_modos_por_energia(sigma, energia_deseada=0.99999)
    print(f"Cantidad de modos para desplazamientos: {n_modos_u}")
    V_reducida = V_svd[:, :n_modos_u]

    print("Reduciendo matrices de rigidez...")
    A_reducidas = [V_reducida.T @ A_sparse @ V_reducida for A_sparse in list_A_scipy]
    f_reducidos = [V_reducida.T @ fv_i for fv_i in list_fv]

    B_vectores = [np.linalg.inv(A_r_i).flatten() for A_r_i in A_reducidas]
    B_matriz = np.array(B_vectores).T
    
    Phi, Sigma_B, _ = np.linalg.svd(B_matriz, full_matrices=False)
    
    # Imprime los valores singulares para entender su magnitud
    print("Valores singulares de la matriz de rigidez (Sigma_B):", Sigma_B)
    
    n_modos_B = contar_modos_por_energia(Sigma_B, energia_deseada=0.99999)
    print(f"Cantidad de modos para matriz de rigidez: {n_modos_B}")
    Phi_reducida = Phi[:, :n_modos_B]

    Thetas = Phi_reducida.T @ B_matriz
    
    return snapshots_matrix, V_reducida, Phi_reducida, training_set, Thetas.T, f_reducidos