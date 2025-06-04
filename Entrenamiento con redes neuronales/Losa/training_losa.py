from dolfin import *
import numpy as np
from multiprocessing import Pool
import random
from tqdm import tqdm
from POD import POD_with_modes

def solve_snapshot(args):
    """Wrapper function to unpack arguments and call the original solve_snapshot."""
    mu, Lx, Ly, nx, ny = args
    # Original solve_snapshot logic
    # Crear la malla del cuadrado
    # Malla
    mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), nx, ny)  

    # Definir elementos finitos y espacio mixto
    V = FiniteElement("P", mesh.ufl_cell(), 1)  # Subimos a grado 3 para mayor estabilidad
    W = FiniteElement("P", mesh.ufl_cell(), 1)  # Igual para w
    M = FunctionSpace(mesh, MixedElement([V, W]))  # Espacio mixto

    # Condiciones de borde
    def boundary(x, on_boundary):
        return on_boundary

    bc_u = DirichletBC(M.sub(0), Constant(0), boundary)  # u = 0
    bc_w = DirichletBC(M.sub(1), Constant(0), boundary)  # w = 0
    bcs = [bc_u, bc_w]

    #Subdominios
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    subdomains.set_all(0)
    # Marcar subcuadrados (2x2): dividimos el cuadrado en 4
    for cell in cells(mesh):
        x, y = cell.midpoint().x(), cell.midpoint().y()
        if x <= Lx/2 and y <= Ly/2:
            subdomains[cell] = 1  # abajo izquierda
        elif x > Lx/2 and y <= Ly/2:
            subdomains[cell] = 2  # abajo derecha
        elif x <= Lx/2 and y > Ly/2:
            subdomains[cell] = 3  # arriba izquierda
        else:
            subdomains[cell] = 4  # arriba derecha

    dx = Measure("dx")(subdomain_data=subdomains)

    # Problema variacional mixto
    (u, w) = TrialFunctions(M)
    (v2, v1) = TestFunctions(M)
    B = 2.747252747e9
    esp= mu[4]*0.28 +0.32

    theta = [0, 0, 0, 0]
    theta[0] = mu[0]*10**4/(esp**3)  # q1
    theta[1] = mu[1]*10**4/(esp**3)  # q2
    theta[2] = mu[2]*10**4/(esp**3)  # q3
    theta[3] = mu[3]*10**4/(esp**3)  # q4

    a = u*v1*dx + inner(grad(w),grad(v1))*dx - inner(grad(u),grad(v2))*dx
    L = theta[0]* v2/B * dx(1)
    for i in range(1,4):
        L+= theta[i]* v2/B * dx(i+1)

    # Resolver
    sol = Function(M)
    solve(a == L, sol, bcs);

    # Extraer u y w
    u, w = sol.split(deepcopy=True)

    #import matplotlib.pyplot as plt
    #plot_obj= plot(u, title= f"sol para {mu}")
    #plt.colorbar(plot_obj, label="Valor de u")
    #plt.show()

    return w.vector().get_local() # Devolver como array NumPy

def training_data(n):
    print("INICIO TRAINING DATA: ", n)

    training_set = [
        tuple(random.uniform(-1, 1) for _ in range(5))  for _ in range(n)
    ]

    # Parámetros del dominio
    Lx, Ly = 5.0, 5.0  # Tamaño del cuadrado
    nx, ny = 32, 32    # Número de divisiones de la malla

    # Crear la malla del cuadrado
    mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)
    V = FunctionSpace(mesh, FiniteElement("P", mesh.ufl_cell(), 1))
    snapshots_matrix = np.zeros((V.dim(), len(training_set)))
    print(snapshots_matrix.shape)
         
    args_for_snapshots = [
        (mu, Lx, Ly, nx, ny)
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

    
    prueba = random.randint(0, n-1)
    prueba_sol = solve_snapshot((training_set[prueba], Lx, Ly, nx, ny))
    res = np.linalg.norm(prueba_sol-snapshots_matrix[:,prueba])
    print("Si este número es cercano a 0, el orden se preserva")
    print(res)

    print("SNAPSHOTS OBTENIDOS")
    U, sigma, Vt = np.linalg.svd(snapshots_matrix, full_matrices=False)

    cant_modos = 0
    for i in range(len(sigma)):
        if sigma[i] > 1e-3:
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