from dolfin import *
import numpy as np
from multiprocessing import Pool
import random
from tqdm import tqdm
from POD import POD_with_modes

def solve_snapshot(args):
    """Wrapper function to unpack arguments and call the original solve_snapshot."""
    mu, mesh_file, facet_file, physical_file, lambda_1, lambda_2 = args
    mesh = Mesh(mesh_file)
    subdomains = MeshFunction("size_t", mesh, physical_file)
    boundaries = MeshFunction("size_t", mesh, facet_file)
    V = VectorFunctionSpace(mesh, "P", 1)

    def left_boundary(x, on_boundary):
        return on_boundary and near(x[0], 0.0)

    f = Constant((1.0, 0.0))
    bc0 = DirichletBC(V, Constant((0.0, 0.0)), left_boundary)
    bcs = [bc0]

    dx = Measure("dx")(subdomain_data=subdomains)
    ds = Measure("ds")(subdomain_data=boundaries)

    u = TrialFunction(V)
    v = TestFunction(V)
    a =  (mu[0]*19/2+21/2) *(2.0 * lambda_2 * inner(sym(grad(u)), sym(grad(v))) + 
                 lambda_1 * tr(sym(grad(u))) * tr(sym(grad(v)))) * dx(1) 

    for i in range(1, 9):
        a += (mu[0]*19/2+21/2)  * (2.0 * lambda_2 * inner(sym(grad(u)), sym(grad(v))) + #Ahora con esto la entrada puede ser entre (0,1)
                      lambda_1 * tr(sym(grad(u))) * tr(sym(grad(v)))) * dx(i+1)

    L = mu[1] * inner(f, v) * ds(2) + mu[2] * inner(f, v) * ds(3) + mu[3] * inner(f, v) * ds(4)

    sol = Function(V)
    solve(a == L, sol, bcs)

    return sol.vector().get_local()

def training_data(n, mesh_file, facet_file, physical_file):
    print("INICIO TRAINING DATA: ", n)

    training_set = [
        #tuple(1 for _ in range(6)) +
        tuple(random.uniform(-1, 1) for _ in range(1)) +
        tuple(random.uniform(-1, 1) for _ in range(3))
        for _ in range(n)
    ]

    mesh = Mesh(mesh_file)
    V = VectorFunctionSpace(mesh, "P", 1)
    snapshots_matrix = np.zeros((V.dim(), len(training_set)))
    print(snapshots_matrix.shape)

    E = 1.0
    nu = 0.3
    lambda_1 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lambda_2 = E / (2.0 * (1.0 + nu))

            
    args_for_snapshots = [
        (mu, mesh_file, facet_file, physical_file, lambda_1, lambda_2)
        for mu in training_set
    ]
            
    # Paralelizar la creación de snapshots con barra de progreso usando imap
    with Pool() as pool:
        results = list(tqdm(
            pool.imap(  # Usar imap para progreso en vivo y mantener el orden
                solve_snapshot,
                args_for_snapshots
            ),
            total=len(training_set),
            desc="Procesando snapshots"
        ))

    # Llenar la matriz de snapshots
    # 'results[j]' corresponde a la solución para 'training_set[j]'
    for j, snapshot in enumerate(results):
        snapshots_matrix[:, j] = snapshot

    
    prueba = random.randint(0, n)
    prueba_sol = solve_snapshot((training_set[prueba], mesh_file, facet_file, physical_file, lambda_1, lambda_2))
    res = np.linalg.norm(prueba_sol-snapshots_matrix[:,prueba])
    print("Si este número es cercano a 0, el orden se preserva")
    print(res)

    print("SNAPSHOTS OBTENIDOS")
    U, sigma, Vt = np.linalg.svd(snapshots_matrix, full_matrices=False)

    cant_modos = 0
    for i in range(len(sigma)):
        if sigma[i] > 1e-1:
            cant_modos += 1
        else:
            break

    cant_modos = min(10, cant_modos)
    print("Cantidad de modos: ", cant_modos)

    reduced_basis = POD_with_modes(snapshots_matrix, cant_modos)
    print("POD LISTA")
    print("VERIFICACION POD TRAINING")
    max_err = 0
    arg_max = -1
    residuals_list = []
    salidas = []
    relativos = []
    for i in range(len(training_set)):
        b = snapshots_matrix[:, i]
        alpha, residuals, rank, s = np.linalg.lstsq(reduced_basis, b, rcond=None)
        salidas.append(alpha)
        relativos.append(residuals[0] / np.linalg.norm(b))
        residuals_list.append(residuals[0])
        if residuals[0] > max_err:
            max_err = residuals[0]
            arg_max = i

    print(max_err)
    print(arg_max)
    print(np.mean(residuals_list))
    print("Error relativo maximo: ", np.max(relativos))
    print("Error relativo medio: ", np.mean(relativos))

    return reduced_basis, salidas, training_set, snapshots_matrix