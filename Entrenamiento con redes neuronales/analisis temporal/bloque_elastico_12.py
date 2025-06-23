from dolfin import *
import time
import random
from scipy.sparse import csr_matrix

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

start_time = time.time()
# 1. Mallado
mesh_file = "data/elastic_block.xml"
facet_file = "data/elastic_block_facet_region.xml"
physical_file = "data/elastic_block_physical_region.xml"

# Generar los parámetros
mu = tuple(random.uniform(-1, 1) for _ in range(12)) 
# Configuración de FEniCS
E, nu = 1.0, 0.3
lambda_1 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
lambda_2 = E / (2.0 * (1.0 + nu))
args_for_snapshots = (mu, mesh_file, facet_file, physical_file, lambda_1, lambda_2)

solve_snapshot(args_for_snapshots)

finish_time = time.time()

print("El bloque elastico con 12 parámetros demora: ", finish_time-start_time)