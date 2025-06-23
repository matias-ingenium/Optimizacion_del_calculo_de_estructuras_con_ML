import numpy as np

#El más recomendado
def contar_modos_por_energia(sigma, energia_deseada=0.9999, max_modos=50):
    """
    Selecciona el número de modos necesarios para capturar un porcentaje
    de la energía total del sistema.
    """
    # La energía es proporcional al cuadrado de los valores singulares
    energia_total = np.sum(sigma**2)
    if energia_total == 0:
        return 1 # Evitar división por cero si todas las soluciones son cero

    energia_acumulada = np.cumsum(sigma**2) / energia_total
    
    # Encontrar el primer índice donde la energía acumulada supera el umbral
    # np.argmax devuelve el índice de la primera ocurrencia de 'True'
    cant = np.argmax(energia_acumulada >= energia_deseada) + 1
    
    return min(cant, max_modos)

# Ejemplo de uso:
# n_modos = contar_modos_por_energia(Sigma_B)



# El que usa en el paper
def contar_modos_relativo(sigma, tol_relativa=1e-3, max_modos=50):
    """
    Selecciona modos basados en una tolerancia relativa al valor singular más grande.
    """
    if len(sigma) == 0 or sigma[0] == 0:
        return 1

    cant = np.sum(sigma / sigma[0] > tol_relativa)
    
    if cant == 0:
        return 1 # Asegurar al menos un modo
        
    return min(cant, max_modos)

# Ejemplo de uso:
# n_modos = contar_modos_relativo(Sigma_B, tol_relativa=1e-3)


#El que veníamos usando (no es escalable)
# --- Función Auxiliar para Contar Modos (Versión Robusta) ---
def contar_modos(sigma, tol=1e-5, max_modos=50):
    """
    Cuenta el número de modos singulares por encima de una tolerancia.
    Garantiza que al menos se devuelva 1 modo si hay valores singulares.
    """
    cant = np.sum(sigma > tol)
    

    if cant == 0 and len(sigma) > 0:
        print(f"Advertencia: Ningún valor singular superó la tolerancia de {tol}. Seleccionando 1 modo por defecto.")
        cant = 1
        
    return min(cant, max_modos)