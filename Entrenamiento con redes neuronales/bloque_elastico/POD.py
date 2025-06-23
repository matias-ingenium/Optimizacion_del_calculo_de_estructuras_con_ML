import numpy as np
from UQpy.dimension_reduction import SnapshotPOD


def POD_with_modes(snapshots_matrix, modes):
    #Pasar la matriz 2d a una 3d
    snapshots_matrix_3d = np.expand_dims(snapshots_matrix, axis=1)
    pod = SnapshotPOD(solution_snapshots=snapshots_matrix_3d, n_modes=modes)
    W= pod.reduced_solution
    # Elimina la dimensión de tamaño 1
    W_reducida = np.squeeze(W, axis=1)
    return W_reducida

def POD_snapshot_matrix(snapshots_matrix, fidelidad):
  #Pasar la matriz 2d a una 3d
  snapshots_matrix_3d = np.expand_dims(snapshots_matrix, axis=1)
  pod = SnapshotPOD(solution_snapshots=snapshots_matrix_3d, reconstruction_percentage=fidelidad)
  W= pod.reduced_solution
  # Elimina la dimensión de tamaño 1
  W_reducida = np.squeeze(W, axis=1)
  return W_reducida