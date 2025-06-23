# Bases-Reducidas


Proyecto de Optimización de Cálculo de Estructuras con FEM, ROM, Redes Neuronales y modelos de ensamble basados en árboles
Descripción General
Este proyecto se enfoca en la optimización del cálculo de estructuras mediante el Método de los Elementos Finitos (FEM) utilizando el Método de la Base Reducida (ROM), Redes Neuronales, Random Forest y Gradient Boosting. Se exploran alternativas para la construcción de la base reducida y la ejecución de la fase online, con un énfasis particular en métodos no intrusivos. También se intenta predecir la inversa de la matriz de rigidez reducida. El código fuente está escrito en Python.

El proyecto se estructura en cuatro componentes principales:

Investigación Teórica: Contiene la fundamentación necesaria para comprender el problema, abarcando el problema físico, el FEM y el ROM con sus diversas alternativas.

Pruebas Iniciales (Google Colab): Documentos con las primeras exploraciones, utilizando FEniCS para FEM y RBniCS para ROM intrusivo. Dada la complejidad de trabajar directamente con las ecuaciones diferenciales, se transita hacia un enfoque no intrusivo empleando Descomposición Ortogonal en Propios (POD) para hallar la base reducida y Redes Neuronales para la etapa online. Se utiliza UQpy para pruebas de POD y ejemplos con redes.

Entrenamiento de Redes Neuronales (Local): Scripts diseñados para ejecución local, donde se implementa el proceso no intrusivo (POD + Red Neuronal) para dos casos de estudio: un bloque elástico y una losa.

Predicción de la matriz inversa (Local): Scripts diseñados para ejecución local, se implementa el proceso no intrusivo para dos variantes del bloque elástico

Estructura de Carpetas y Archivos
1. Investigación Teórica
└── Marco Teórico/
    └── Marco_teorico.ipynb (Documento Jupyter para abrir con Google Colab)

2. Documentos en Google Colab
├── Fenics/
│   ├── Kirchhoff-Love_plate_fenics.ipynb (Formulación débil y solución FEM en FEniCS para la losa de Kirchhoff-Love. Incluye comparación de resultados.)
│   └── Solucion_barra_Euler_fenics.ipynb (Pruebas iniciales para validar la formulación débil y su implementación con FEniCS para la barra de Euler.)
│
├── Rbnics/
│   ├── Barra_Euler_rbnics.ipynb (Deducción y formulación débil de la ecuación de la barra de Euler. Aplicación de ROM con RBniCS y comparación de resultados.)
│   ├── Copia_de_tutorial_elastic_block_comentado.ipynb (Ejemplo resuelto de un bloque elástico con rigidez variable usando RBniCS, con comentarios detallados.)
│   ├── Presentacion_Losa.ipynb (Versión organizada y visual del problema de la losa de Kirchhoff-Love para presentaciones.)
│   └── Kirchhoff-Love_plate_RBniCS.ipynb (Resolución del problema de la losa con ROM usando RBniCS, variando fuerzas y espesor.)
│
└── Redes/
    ├── Elementos_para_red_con_rigidez/ (Archivos para entrenar la red del bloque elástico)
    │   ├── training_set.dat
    │   ├── testing_set.dat
    │   ├── snapshots_matrix.dat
    │   ├── base_reducida.dat
    │   ├── salidas.dat
    │   ├── salidas_esperadas.dat
    │   └── result_matrix.dat
    └── Generalizacion_redes.ipynb (Explicación del entrenamiento de la red para predecir coeficientes. Requiere ejecución previa de "Elementos_para_red_con_rigidez".)

3. Entrenamiento con Redes Neuronales (Ejecución Local)
Esta sección contiene los scripts para la ejecución local de los experimentos principales de predicción de campos de desplazamiento. Se prueban diversas técnicas además de redes neuronales, como regresión polinómica, Random Forest y otros modelos lineales.

Requisitos de Bibliotecas Principales:

FEniCS (versión 2019, no FEniCSx)

UQpy

Pytorch

Scikit-learn (Sklearn)

Otras bibliotecas estándar de Python (Numpy, Matplotlib, etc.)

Experimentos:

A. Bloque Elástico
Descripción: Un cuadrado dividido en 9 subcuadrados, empotrado en el lado izquierdo, con fuerzas aplicadas en 3 secciones del lado derecho. (Ver imagen en anexos del documento principal).
├── Bloque_Elastico/
│   ├── Main.py (Script principal: carga/crea datos, entrena y evalúa la red.)
│   ├── Mallado.py (Funciones para importar archivos de mallado.)
│   ├── POD.py (Funciones para implementar POD con UQpy.)
│   ├── Polinomio.py (Implementación de regresión polinómica.)
│   ├── Prueba_test/ (Directorio con datos de testeo)
│   ├── Prueba_train/ (Directorio con datos de entrenamiento)
│   ├── Redes.py (Clase para el manejo de redes neuronales con Pytorch.)
│   ├── Testing.py (Script para generar datos de testeo.)
│   ├── Training.py (Script para generar datos de entrenamiento.)
│   └── Training2.py (Script para generar datos de entrenamiento en paralelo - recomendado.)

B. Losa
Descripción: Losa cuadrada dividida en 4 cuadrantes, fija en los bordes. Se aplica una fuerza en cada cuadrante y se varía el espesor. (Ver imagen en anexos del documento principal). La estructura de archivos es similar al experimento del bloque elástico.
├── Losa/
│   ├── Elementos_de_prueba/ (Directorio con conjuntos de entrenamiento y testeo.)
│   ├── Lineales.py (Batería de pruebas con métodos lineales.)
│   ├── Lineales2.py (Similar al anterior.)
│   ├── Main.py (Script principal: carga/crea datos, entrena y evalúa.)
│   ├── POD.py (Funciones para implementar POD con UQpy.)
│   ├── Polinomio.py (Implementación de regresión polinómica.)
│   ├── Random_Forest.py (Implementación de Random Forest para aproximar multiplicadores.)
│   ├── Redes.py (Clase para el manejo de redes neuronales con Pytorch.)
│   ├── Redes_paralelo.py (Script para probar redes de distintas arquitecturas simultáneamente.)
│   ├── Relaciones.py (Script para visualizar relaciones entre coordenadas de entrada y salida.)
│   ├── SVR.py (Implementación de Support Vector Regression.)
│   ├── Training_losa.py (Script para generar datos de entrenamiento para la losa.)
│   └── Testing_losa.py (Script para generar datos de testeo para la losa.)


4. Generando Matriz Inversa
Descripción: Se predice la inversa de la matriz de rigidez reducida utilizando POD, random forest y gradient boosting (no se predice el vector de cargas en los ejemplos).
├── Losa/
│   ├── Data/ (Contiene los datos del mallado para importar)
│   ├── Elementos_de_prueba/ (Directorio con conjuntos de entrenamiento y testeo.)
│   ├── Cant_modos.py (Contiene distintas formas de contar la cantidad de modos en la base reducida con SVD.)
│   ├── Fase_online.py (Realiza la fase online para cierto parámetro)
│   ├── Ku_f.py (Muestra como en fenics se puede obtener el sistema Ku=f)
│   ├── Main.py (Script principal: carga/crea datos, entrena y evalúa, puede ser utilizando Gradient Boosting o Random Forest.)
│   ├── Mallado.py (para descargar los datos del mallado)
│   ├── prediccion_f.py (Muestra como se puede predicr f con regresión lineal para este modelo)
│   ├── Testing_set.py (genera datos de prueba para el bloque elástico con 12 parámetros)
│   ├── Testing_set_simplificado.py (genera datos de prueba para el bloque elástico con 4 parámetros)
│   ├── Training_set.py (genera datos de entrenamiento para el bloque elástico con 12 parámetros)
│   ├── Training_set_simplificado.py (genera datos de entrenamiento para el bloque elástico con 4 parámetros)

Este README proporciona una visión general de la estructura y contenido del proyecto. Para detalles específicos sobre la metodología, resultados y conclusiones, por favor refiérase al documento de investigación principal.
