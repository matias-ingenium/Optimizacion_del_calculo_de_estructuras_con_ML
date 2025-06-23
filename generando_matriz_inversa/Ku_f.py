from dolfin import *

def solve_snapshot(mu):
    # Parámetros del dominio
    Lx, Ly = 5.0, 5.0  # Tamaño del cuadrado
    nx, ny = 32, 32    # Número de divisiones de la malla
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

# --- INICIO DE LA MODIFICACIÓN ---

    # Ensamblar la matriz A y el vector f
    A, f = assemble_system(a, L, bcs)
    return A, f
    
    
    print("Dimensiones de la matriz de rigidez (A):", A.size(0), "x", A.size(1))
    print("Dimensión del vector del lado derecho (f):", f.size())

    # Se puede convertir la matriz a un formato denso de NumPy si es necesario (cuidado con el tamaño)
    # import numpy as np
    # A_np = A.array()
    # f_np = f.get_local()

    # --- FIN DE LA MODIFICACIÓN ---

A, f = solve_snapshot((1,1,1,1,0.3))
print(A)
print(f)