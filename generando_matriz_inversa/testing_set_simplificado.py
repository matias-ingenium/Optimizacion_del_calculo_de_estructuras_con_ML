# training_set.py (CORRECCIÓN PARA EVITAR 0 MODOS)
from dolfin import *
import numpy as np
from multiprocessing import Pool
import random
from tqdm import tqdm
from scipy.sparse import csr_matrix
import pickle
import os

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
        a += (mu[0]*19/2+21/2)  * (2.0 * lambda_2 * inner(sym(grad(u)), sym(grad(v))) + #Ahora con esto la entrada puede ser entre (0,1)
                      lambda_1 * tr(sym(grad(u))) * tr(sym(grad(v)))) * dx(i+1)

    f_load = Constant((1.0, 0.0))
    L = mu[1] * inner(f_load, v) * ds(2) + mu[2] * inner(f_load, v) * ds(3) + mu[3] * inner(f_load, v) * ds(4)
    
    A_dolfin, fv = assemble_system(a, L, [bc])
    sol = Function(V)
    solve(A_dolfin, sol.vector(), fv)
    return sol.vector().get_local(), fv.get_local()


def generate_test_data(n_test_snapshots, mesh_file, facet_file, physical_file):
    """
    Genera un conjunto de datos de testeo. Carga los datos desde un archivo
    si ya existen, o los calcula y guarda si no.
    """
    print(f"\nINICIO DE GENERACIÓN DE DATOS DE TESTEO: {n_test_snapshots} snapshots")
    
    # 1. Definir el nombre del archivo de guardado
    test_data_file = f"generando_matriz_inversa/elementos de prueba/datos_testeo_simplificado_{n_test_snapshots}.pkl"

    # 2. Comprobar si el archivo de datos ya existe
    if os.path.exists(test_data_file):
        print(f"Cargando datos de testeo pre-calculados desde '{test_data_file}'...")
        with open(test_data_file, "rb") as f:
            # Cargar la tupla completa
            test_data_tuple = pickle.load(f)
        print("Carga completada.")
        
        # Devolver los datos cargados
        return test_data_tuple
    
    else:
        print(f"No se encontraron datos de testeo guardados. Generando {n_test_snapshots} nuevos snapshots...")
        
        # --- Bloque de cálculo pesado ---
        
        # Generar nuevos parámetros aleatorios para el testeo
        test_set_params = [tuple(random.uniform(-1, 1) for _ in range(4)) for _ in range(n_test_snapshots)]
        
        # Configuración de FEniCS
        E, nu = 1.0, 0.3
        lambda_1 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        lambda_2 = E / (2.0 * (1.0 + nu))
        args_for_snapshots = [(mu, mesh_file, facet_file, physical_file, lambda_1, lambda_2) for mu in test_set_params]
        
        # Ejecutar el cálculo en paralelo
        with Pool() as pool:
            results = list(tqdm(pool.imap(solve_snapshot, args_for_snapshots), total=len(test_set_params), desc="Procesando snapshots de testeo"))

        # Desempaquetar los resultados
        test_solutions, test_fvs = zip(*results)
        
        # Organizar los datos para guardarlos y devolverlos
        final_params = np.array(test_set_params)
        final_solutions = np.array(test_solutions).T
        final_fvs = list(test_fvs) # Guardar como lista
        
        data_to_save = (final_params, final_solutions, final_fvs)

        # Guardar la tupla completa
        print(f"Guardando datos de testeo en '{test_data_file}'...")
        with open(test_data_file, "wb") as f:
            pickle.dump(data_to_save, f)
        print("Guardado completado.")

        print("GENERACIÓN DE DATOS DE TESTEO COMPLETADA")
        return data_to_save