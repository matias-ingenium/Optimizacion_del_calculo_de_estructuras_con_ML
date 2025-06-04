import os
import requests
from dolfin import *

def mallado():

    # Crear el directorio 'data' si no existe
    os.makedirs("data", exist_ok=True)

    # Lista de URLs y nombres de archivos
    files = [
        (
            "https://github.com/RBniCS/RBniCS/raw/master/tutorials/02_elastic_block/data/elastic_block.xml",
            "data/elastic_block.xml",
        ),
        (
            "https://github.com/RBniCS/RBniCS/raw/master/tutorials/02_elastic_block/data/elastic_block_facet_region.xml",
            "data/elastic_block_facet_region.xml",
        ),
        (
            "https://github.com/RBniCS/RBniCS/raw/master/tutorials/02_elastic_block/data/elastic_block_physical_region.xml",
            "data/elastic_block_physical_region.xml",
        ),
    ]

    # Descargar cada archivo si no existe
    for url, filepath in files:
        if not os.path.exists(filepath):
            print(f"Descargando {filepath}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"{filepath} descargado exitosamente.")
            else:
                print(f"Error al descargar {filepath}: {response.status_code}")
        else:
            print(f"{filepath} ya existe, omitiendo descarga.")

    # Cargar la malla y las funciones de malla
    mesh = Mesh("data/elastic_block.xml")
    subdomains = MeshFunction("size_t", mesh, "data/elastic_block_physical_region.xml")
    boundaries = MeshFunction("size_t", mesh, "data/elastic_block_facet_region.xml")

    V = VectorFunctionSpace(mesh, "Lagrange", 1) #Grado de la interpolación

    return mesh, subdomains, boundaries, V