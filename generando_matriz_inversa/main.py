# main.py
import os
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestRegressor
from fase_online import fase_online

"""
#Para la versión simplificada (4 parámetros)
from training_set_simplificado import training_data 
from testing_set_simplificado import generate_test_data 
"""
#Para la versión con 12 parámetros
from training_set import training_data 
from testing_set import generate_test_data 

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m" 
RESET = "\033[0m"

# --- FASE DE ENTRENAMIENTO ---
print(f"{YELLOW}--- INICIANDO FASE DE ENTRENAMIENTO ---{RESET}")
training_file = "generando_matriz_inversa/elementos de prueba/modelo_rom_entrenado_complejo_1.pkl"
n_train_snapshots = 1000

if os.path.exists(training_file):
    print("Cargando modelo ROM y ML entrenado desde archivo...")
    with open(training_file, "rb") as f:
        V_reducida, Phi_reducida, modelo_ml = pickle.load(f)
else:
    # 1. Generar datos de entrenamiento y construir el ROM
    mesh_file = "data/elastic_block.xml"
    facet_file = "data/elastic_block_facet_region.xml"
    physical_file = "data/elastic_block_physical_region.xml"
    
    start_time= time.time()
    # Esta función viene de 'training_set.py' y debe devolver todo lo necesario
    _, V_reducida, Phi_reducida, mu_train, ThetasT_train, _ = training_data(
        n_train_snapshots, mesh_file, facet_file, physical_file
    )
    end_time= time.time()
    print(f"{GREEN}Duración de training_data: {(end_time - start_time):.2f} segundos{RESET}")

    # 2. Entrenar el modelo de Machine Learning
    """
    print("\nEntrenando el modelo de Machine Learning (Random Forest)...")
    modelo_ml = RandomForestRegressor(
    n_estimators=4000, 
    max_depth=None,          # Le permite al árbol crecer tan profundo como quiera.
    min_samples_split=2,     # Permite divisiones incluso en los nodos más pequeños.
    min_samples_leaf=1,      # Permite que las hojas sean ultra-específicas.
    random_state=77
    )
    
    print("Entrenamiento completado.")
    """
    # Importa el envoltorio MultiOutputRegressor
    from sklearn.multioutput import MultiOutputRegressor
    import lightgbm as lgb

    # --- 4. ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING ---
    # Gradient boosting
    print("\nEntrenando el modelo de Machine Learning (LightGBM con MultiOutputRegressor)...")

    # a) Primero, define tu modelo base LGBMRegressor con sus hiperparámetros
    #    Este es el modelo que se usará para cada una de las salidas.
    lgbm_base = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        num_leaves=15,
        random_state=77,
        n_jobs=-1
    )

    # b) Luego, envuelve el modelo base con MultiOutputRegressor
    #    Este es el objeto que realmente se entrena.
    modelo_ml = MultiOutputRegressor(lgbm_base)
    
    # c) Ahora, el .fit() funciona perfectamente con tu matriz de objetivos múltiples
    start_time= time.time()
    modelo_ml.fit(mu_train, ThetasT_train)
    end_time = time.time()
    print(f"{GREEN}Duración del GB: {(end_time - start_time):.2f} segundos{RESET}")
    # Calcular el puntaje R² en los mismos datos de entrenamiento
    score_entrenamiento = modelo_ml.score(mu_train, ThetasT_train)

    print(f"Entrenamiento completado.")
    print(f"Puntaje R² en el conjunto de entrenamiento: {score_entrenamiento:.4f}")

    # 'estimators_' es una lista con un modelo LGBM para cada objetivo Theta
    # Por ejemplo, veamos la importancia de las características para el primer coeficiente Theta
    importancias_theta_0 = modelo_ml.estimators_[0].feature_importances_
    print(f"Importancia de las características para predecir el primer Theta: {importancias_theta_0}")

    # También se puede promediar la importancia a través de todos los modelos
    importancias_promedio = np.mean([est.feature_importances_ for est in modelo_ml.estimators_], axis=0)
    print(f"Importancia promedio de las características en todos los objetivos: {importancias_promedio}")




# --- FASE DE TESTEO ---
print(f"\n{YELLOW}--- INICIANDO FASE DE TESTEO ---{RESET}")
n_test_snapshots = 40

# 4. Generar un conjunto de datos de testeo completamente nuevo
mu_test, u_test_true, f_test_true = generate_test_data(
    n_test_snapshots, "data/elastic_block.xml", "data/elastic_block_facet_region.xml", "data/elastic_block_physical_region.xml"
)

# 5. Evaluar el modelo en el conjunto de testeo
print(f"\nEvaluando el modelo en {n_test_snapshots} muestras de testeo...")
errores_relativos_test = []
start_time = time.time()
for i in range(n_test_snapshots):
    # Solución de referencia (verdadera) del resolvedor FEniCS
    u_referencia = u_test_true[:, i]
    
    # Parámetros del caso de testeo actual
    mu_actual = mu_test[i]
    
    # Vector de fuerza verdadero para este caso de testeo
    f_vector_true = f_test_true[i]

    # Proyectar el vector de fuerza al espacio reducido USANDO LA BASE DEL ENTRENAMIENTO (V_reducida)
    f_r_actual = V_reducida.T @ f_vector_true
    
    # Predecir la solución usando el modelo ROM-ML
    u_predicho = fase_online(mu_actual, V_reducida, Phi_reducida, modelo_ml, f_r_actual)
    
    # Calcular y guardar el error
    error_absoluto = np.linalg.norm(u_referencia - u_predicho)
    norma_referencia = np.linalg.norm(u_referencia)
    
    if norma_referencia > 1e-10:
        errores_relativos_test.append((error_absoluto / norma_referencia) * 100)
end_time = time.time()
print(f"{GREEN}Duración de etapa online promedio: {(end_time - start_time)/n_test_snapshots} segundos{RESET}")
if errores_relativos_test:
    error_promedio_test = np.mean(errores_relativos_test)
    error_maximo_test = np.max(errores_relativos_test)
    print(f"{GREEN}Error relativo PROMEDIO en el CONJUNTO DE TESTEO: {error_promedio_test:.4f}%{RESET}")
    print(f"{GREEN}Error relativo MÁXIMO en el CONJUNTO DE TESTEO: {error_maximo_test:.4f}%{RESET}")


# --- 6. GUARDADO OPCIONAL DEL MODELO ENTRENADO ---
print("\n" + "="*40)
print("Fase de entrenamiento y testeo completada.")

# Pide confirmación al usuario antes de guardar
# Usamos .lower() para aceptar 's', 'S', 'si', 'SI', etc.
# Usamos .strip() para eliminar espacios en blanco accidentales
respuesta = input(f"¿Desea guardar este modelo entrenado en '{training_file}'? (Esto sobrescribirá el archivo) [s/n]: ")

if respuesta.lower().strip().startswith('s'):
    try:
        # Este es tu bloque de código original para guardar
        print(f"Guardando modelo en '{training_file}'...")
        with open(training_file, "wb") as f:
            pickle.dump((V_reducida, Phi_reducida, modelo_ml), f)
        print(f"{GREEN}¡Éxito! Modelo guardado.{RESET}")
    except Exception as e:
        print(f"{RED}Error: No se pudo guardar el modelo. {e}{RESET}")
else:
    print("Guardado del modelo omitido por el usuario.")

print("="*40)