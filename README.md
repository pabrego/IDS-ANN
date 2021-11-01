# IDS-ANN
# Tarea 3A
El objetivo es mplementar y evaluar un sistema de detección de intrusos. Para esto se deben realizar lo siguientes pasos:
- Calibrar los **pesos de salida** de la red neuronal usando el algoritmo pseudo-inversa  **Moore-Penrose**.
- Calibrar **pesos ocultos** de la red neuronal con el algoritmo de optimización **PSO**
- Evaluar el rendimiento de la IDS con con las siguientes métricas:
	-	[[Matriz de confusión]]
	-	Fscore


## Formato y estructura de archivos
Se entrega un archivo `config_ann.csv` con la siguiente estructura:
- Lines 1: número máximo de iteraciones
- Linea 2: número de nodos ocultos
- Linea 3: número de partículas
- Linea 4: penalidad p-inversa

Estas lineas se cargan en un array llamado `param[0-3]`

---

Se entrega archivos con la data de training y testing `train.csv` y `test.csv` con la siguiente estructura:
- 42 columnas de características (variables explicativas)
- 1 columna de salida, con vaores 1 y -1, que representan si el registro es o no un intruso en la red

---

El archivo `trainp.py` contiene la función `train_pso` donde se inicializan los valores del algorito PSO, y se corren las iteraciones indicadas en `param[0]` .

El archivo test.py contiene la función donde :
- carga los datos de testing (datos de entrada `xv` y salida `yv` )
- cargan los pesos de la red ya entrenada
- se ejecuta la red (`ut.forward(datos de entrada, pero1, peso2)`)
- se generan y exportan las métricas (matriz de confusión)

---

El archivo `my_utility.py` contiene las funiones que se debe rellenar. Su desarrollo se explicará en el paso a paso


## Paso a paso de la tarea

Se describirán las pasos que se deben realizar para el desarrollo del proyecto

### Limpieza y preparación de datos
En primer lugar, se deben cargar los datos dentro de la funcion `load_data(file name)` que recibe el nombre del archivo a cargar y devuelve un array de numpy  con los valores de entrada `x` y valores de salida `y`.

Luego, se deben normalizar las columnas de los valores de entrada con la siguiente fórmula:

<img src="https://render.githubusercontent.com/render/math?math=x = \frac{x - x_{min}}{x_{max} - x_{min}} \cdot (b-a) + a">


Donde:
- x -> registro actual
- x_{min} -> valor menor de la columna
- x_{max} -> valor mayor de la columna
- a = 0,01
- b = 0,99
- 
Con los valores normalizados, quedan preparados para empezar la etapa de training de la red neuronal.
