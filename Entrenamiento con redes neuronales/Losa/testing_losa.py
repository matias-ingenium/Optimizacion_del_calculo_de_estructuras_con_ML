from dolfin import *

def testing_data(n, reduced_basis):

    
    print("INICIO TESTING DATA")
    import numpy as np


    import random

    training_set = [
        tuple(random.uniform(-1, 1) for _ in range(5))  for _ in range(n)
    ]

    # Parámetros del dominio
    Lx, Ly = 5.0, 5.0  # Tamaño del cuadrado
    nx, ny = 17, 17    # Número de divisiones de la malla

    # Crear la malla del cuadrado
    mesh = RectangleMesh(Point(0.0, 0.0), Point(Lx, Ly), nx, ny)
      

    # Definir elementos finitos y espacio mixto
    V = FiniteElement("P", mesh.ufl_cell(), 1)  
    W = FiniteElement("P", mesh.ufl_cell(), 1)  
    M = FunctionSpace(mesh, MixedElement([V, W]))  # Espacio mixto

    snapshots_matrix = np.zeros((M.dim()//2, len(training_set)))
    print(snapshots_matrix.shape)

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
    j=0
    for mu in training_set:


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

        print(f"\033[1;37;44m  Paso: {j}  \033[0m")  # Texto blanco con fondo azul

        snapshots_matrix[:,j]=w.vector().get_local()
        j+=1

    print("SNAPSHOTS OBTENIDOS")

    print("Forma snapshots: ", snapshots_matrix.shape)
    print(type(reduced_basis))
    print("Forma Base reducida: ", reduced_basis.shape)
    salidas= []
    relativos=[]
    for i in range(len(training_set)):
        # Vector objetivo
        b = snapshots_matrix[:, i]

        # Resolver por mínimos cuadrados
        alpha, residuals, rank, s = np.linalg.lstsq(reduced_basis, b, rcond=None)
        salidas.append(alpha)
        relativos.append(residuals[0]/np.linalg.norm(b))



    print("VERIFICACION POD TESTING")
    print("Error relativo maximo: ", np.max(relativos))
    print("Error relativo medio: ", np.mean(relativos))

    return salidas, training_set, snapshots_matrix