import numpy as np
def fase_online(mu_nuevo, V_reducida, Phi_reducida, modelo_ml, f_r):
    """
    Predice la solución para un nuevo conjunto de parámetros 'mu_nuevo'.
    """
    
    # --- Paso 1: Predecir los coeficientes Theta con el modelo ML ---
    theta_predicho = modelo_ml.predict([mu_nuevo]) # 
    
    # --- Paso 2: Reconstruir el vector de la inversa de la matriz de rigidez ---
    B_predicho_vector = Phi_reducida @ theta_predicho.T # 
    
    # --- Paso 3: Reconstruir la matriz de rigidez reducida inversa ---
    n = V_reducida.shape[1]
    A_r_inv_predicha = B_predicho_vector.reshape((n, n)) 
    
    # --- Paso 4: Resolver el sistema reducido ---
    # xi = A_r_inv * f_r
    xi_predicho = A_r_inv_predicha @ f_r  
    
    # --- Paso 5: Reconstruir la solución de desplazamiento global ---
    u_predicho = V_reducida @ xi_predicho 
    
    print("Fase Online completada.")
    
    return u_predicho