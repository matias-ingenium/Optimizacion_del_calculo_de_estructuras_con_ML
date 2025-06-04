from dolfin import *
from POD import *
def training_data(n, subdomains, boundaries, V):

    
    print("INICIO TRAINING DATA")
    import numpy as np


    import random

    training_set = [
        tuple(random.uniform(1, 100) for _ in range(9)) +
        tuple(random.uniform(-1, 1) for _ in range(3))
        for _ in range(n)
    ]

    snapshots_matrix = np.zeros((V.dim(), len(training_set)))
    print(snapshots_matrix.shape)

    # Condiciones de borde
    def left_boundary(x, on_boundary):
        return on_boundary and near(x[0], 0.0)

    f = Constant((1.0, 0.0))
    E = 1.0 #Esto es mu_p (vendría a ser el valor de la densidad de el subloque 9?)
    nu = 0.3
    lambda_1 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lambda_2 = E / (2.0 * (1.0 + nu))


    bc0 = DirichletBC(V, Constant((0.0, 0.0)), left_boundary)
    bcs = [bc0]

    # Definir medida para integración
    dx = Measure("dx")(subdomain_data=subdomains)
    ds = Measure("ds")(subdomain_data=boundaries) #ds es para integrar solo en el borde del dominio
    j=0
    for mu in training_set:


        # Problema variacional mixto
        u = TrialFunction(V)
        v = TestFunction(V)
        a = mu[0]*(2.0 * lambda_2 * inner(sym(grad(u)), sym(grad(v))) + lambda_1 * tr(sym(grad(u))) * tr(sym(grad(v)))) * dx(1)
        for i in range(1,9):
            a += mu[i]*(2.0 * lambda_2 * inner(sym(grad(u)), sym(grad(v))) + lambda_1 * tr(sym(grad(u))) * tr(sym(grad(v)))) * dx(i+1)

        L = mu[9]*inner(f,v)* ds(2) + mu[10]*inner(f,v)* ds(3) + mu[11]*inner(f,v)*ds(4)

        # Resolver
        sol = Function(V)
        solve(a == L, sol, bcs);

        print(f"\033[1;37;44m  Paso: {j}  \033[0m")  # Texto blanco con fondo azul

        snapshots_matrix[:,j]=sol.vector().get_local()
        j+=1

    print("SNAPSHOTS OBTENIDOS")
    U, sigma, Vt = np.linalg.svd(snapshots_matrix, full_matrices=False)

    
    cant_modos=0
    for i in range(len(sigma)):
        if sigma[i]>1e-1:
            cant_modos+=1
        else:
            break
        print("Cantidad de modos: ", cant_modos)

    reduced_basis = POD_with_modes(snapshots_matrix, cant_modos)
    print("POD LISTA")
    print("VERIFICACION POD TRAINING")
    max_err=0
    arg_max=-1
    residuals_list=[]
    salidas= []
    relativos=[]
    for i in range(len(training_set)):
        # Vector objetivo
        b = snapshots_matrix[:, i]

        # Resolver por mínimos cuadrados
        alpha, residuals, rank, s = np.linalg.lstsq(reduced_basis, b, rcond=None)
        salidas.append(alpha)
        relativos.append(residuals[0]/np.linalg.norm(b))
        residuals_list.append(residuals[0])
        if residuals[0]>max_err:
            max_err=residuals[0]
            arg_max=i

    print(max_err)
    print(arg_max)
    print(np.mean(residuals_list))
    print("Error relativo maximo: ", np.max(relativos))
    print("Error relativo medio: ", np.mean(relativos))

    return reduced_basis, salidas, training_set, snapshots_matrix