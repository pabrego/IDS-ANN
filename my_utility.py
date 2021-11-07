# My Utility : auxiliars functions
import pandas as pd
import numpy as np
from pprint import pprint


# Initialize  Swarm
def iniSwarm(Nh, Np, d):
    """
    Parameters:
        d: Largo de una fila de X (Cantidad de caracteristicas)
    """
    # Matriz con el swarm inicial
    X = np.zeros((Np, (Nh*d)))
    # Se llena la matriz con pesos aleatoreos
    for i in range(Np):
        X[i] = randW(Nh, d)

    return(X)


# Initialize random weights
def randW(next, prev):
    r = np.sqtr(6/(next + prev))
    w = (np.random.random((next, prev)) * 2 * r) - r
    return(w)

# MLP-fitness


# MLP-fitness
def mlp_Fitness(x, y, X, C):
    """
    x -> Filas = Cantidad de muestras (R)
      -> Columnas = Variables (d)     

    y -> Vector de etiquetas (R)

    X(Matriz Swarm) -> Filas = Numero de particulas (m)
                    -> Columnas = Numero de nodos ocultos (Nh) * Caracteristicas (d)

    funcion objetivo: recibe x y X (variables y pesos (x, W1))

    1. por cada fila de x, 42 variables, las ingreso a la funcion objetivo junto con el enjambre X.
        - 
        - 
        - 
        - Retorno el enjambre optimizado

    """
#   W = [d * NH] primeros d registros corresponden a los pesos que recibe la primera neurona
    d = x.shape[1]      # cantidad de variables
    Nh = X.shape[1]/d   # cantidad de nodos ocultos (neuronas)

    MSE = 0
    SE = []
    # transformo fila de pesos en matriz de d * Nh
    W = np.array(X[0]).reshape(d, Nh)
    # en X[0] almaceno la mejora partícula.

    for i in range(x.shape[0]):                 # por cada registro, calculo un MSE
        r = x[i]
        # array con la salida de cada neurona.
        H = np.zeros(Nh)
        for j in range(Nh):                 # por cada neurona
            # cada valor en H es un vector de largo d
            H.append(act_function(r, W[j]))
        W2 = w2_pinv(H, y[i], C)

    MSE = np.average(SE)

    return(W2, MSE)


# Update Particle fitness
def upd_pFitness():
    # complte
    return(...)


# Update Velocity of PSO
def upd_veloc(P, X, V):
    """
    Esta funcion debe actualizar la velocidad para todo V en funcion de X
    Parameters:
    V: Matriz de velocidad
    X: Matriz swarm
    P['pos']: Mejor swarm
    P['gBest']: Mejor particula
    """
    # Constantes
    c1 = 1.05
    c2 = 2.95
    r1 = np.random.randint(0, 1)
    r2 = np.random.randint(0, 1)
    # Inercia
    '''Imax: iteracion maxima; Iact: iteracion actual'''
    for i in range(len(X)):
        a = 0.95 - (((0.95 - 0.1) / len(X)) * i)
        # Velocidad (V)
        V[i] = a * V[i] + (c1*r1) * (P['pos'][i] + X[i]) + \
            (c2*r2) * (P['gBest'] - X[i])

    return(V)


'''
# Update Swarm of QPSO
def upd_swarm():
    # complete
    return(...)
'''

# Update W2 via P-inverse


def w2_pinv(H, y, C):
    """
    Parameters:
        H: Matriz de nodos ocultos
        y: Datos deseados
        C: Parametro de penalidad de pseudo-inversa
    Returns:
        w2: Peso actualizado
    """
    # Matriz transpuesta de H
    Ht = np.transpose(H)
    # Matriz transpuesta de y
    Yt = np.transpose(y)
    # Dimensiones de la matriz H (se asume que es cuadrada)
    m = np.shape(H)
    # Se calcula B (ver primer parentisis de la formula) para luego calcular la matriz
    # Pseudoinversa de Moore-Penrose
    B = (H * Ht) + (np.identity(m[0]) / C)
    # Matriz transpuesta de B
    Bt = np.transpose(B)
    Binv = np.linalg.inv(Bt * B) * Bt
    # Peso actualizado

    # H * W2 = y
    # W2 = Pseudoinversa * H * Y

    w2 = Binv * H * Yt
    return(w2)


# Feed-forward of ANN
def forward(xv, w1, w2):
    """
    Funcion para probar la red neuronal
    Recibe los valores de entrada de testing, junto con los pesos devueltos en la etapa de training

    Devuelve la salida de la red neuronal: lista con 1 valor por cada registro
    """

    return(z)


def act_function(w, X):
    """
    Funcion de activacion utilizada para obtener la matriz H
    Funcion del Grupo 2: Tangente hiperbolica

    Parameters:
    w: Matriz de pesos
    X: Matriz de datos

    Returns:
        h: Vector de W1
    """
    # complete h(w, X)
    h = np.exp(w * X) - np.exp((-1 * w) * X)
    h /= np.exp(w * X) + np.exp((-1 * w) * X)

    return(h)

# Measure


def metrica(y, z):
    """
    Aqui se utilizan las metricas para evaluar el algoritmo
    """
    conf_m = confusion_matrix(y, z)

    # Confusion matrix to CSV
    pd.DataFrame(conf_m).to_csv("cmatrix_ann.csv")

    VP = conf_m[0][0]  # Verdadero Positivo
    TN = conf_m[1][1]  # Verdadero Negativo
    FP = conf_m[0][1]  # Falso Positivo
    FN = conf_m[1][0]  # Falso negativo

    P = VP / (VP + FP)  # Precision
    R = VP / (VP + FN)  # Sensibilidad
    fScore = 2*((P * R) / (P + R))
    A = (VP + TN) / (VP + TN + FP + FN)  # Exactitud

    # FScore to CSV
    fscoreCsv = pd.DataFrame([fScore], columns=['FScore'])
    fscoreCsv.to_csv('fscore.csv', index=False, sep=',')

    return(...)


# Confusion matrix
def confusion_matrix(y, z):
    """
    Parameters:
        y: Vector con valores esperados
        z: Vector con los resultados del algoritmo

    Returns:
        conf_m: Matriz de confusion
            conf_m[0][0]: True positive
            conf_m[0][1]: False positive
            conf_m[1][0]: False negative
            conf_m[1][1]: True negative

    Nota: Esto funciona asumiendo que Z es un vector
    """
    conf_m = np.zeros((2, 2))

    for i in range(len(y)):
        if y[i] == z[i]:
            if y[i] == -1:
                conf_m[1][1] = conf_m[1][1] + 1  # True negative
            else:
                conf_m[0][0] = conf_m[0][0] + 1  # True positive
        else:
            if y[i] == -1:
                conf_m[1][0] = conf_m[1][1] + 1  # False negative
            else:
                conf_m[0][1] = conf_m[0][0] + 1  # False positive

    return(conf_m)


# Norm of the data
def norm_data(xt):
    """
    Esta funcion se encarga de normalizar los datos.
    Parameters: 
        xt: Matriz con los datos de entrenamiento transpuesta
    Returns:
        xn: Parametro x normalizado
    """
    # Constantes
    a = 0.01
    b = 0.99

    # Es mas facil normalizar todo el vector
    for i in range(len(xt)):
        # Normalizar todo el vector
        col = xt[i]
        # Proceso de normalizacion
        val1 = col - np.amin(col)
        val2 = np.amax(col)-np.amin(col)

        aux = np.divide(val1, val2)
        col = aux*(b-a)+a

        # Se reemplaza la fila original con la normalizada
        xt[i] = col

    # Dehacemos la transposicion
    xn = np.transpose(xt)

    return (xn)

# ------------------------------------------------------------------------
#      LOAD-SAVE
# -----------------------------------------------------------------------


# Configuration of the ANN
def load_config():
    param = np.genfromtxt("Datos/config_ann.csv", delimiter=',', dtype=None)
    par = []
    par.append(np.int16(param[0]))  # Max. Iterations
    par.append(np.int16(param[1]))  # Number of nodes
    par.append(np.int16(param[2]))  # Number of particle
    par.append(np.float(param[3]))  # Penalidad C of the Pinv.
    return(par)


# Load data
def load_data(fname=""):

    fname = 'D:\\Escritorio\\Codigo\\Python\\TareaIDS\\IDS-ANN\\train.csv'
    # Constantes
    x = pd.read_csv(fname, header=None)
    x = np.array(x)

    # Transponemos la matriz para facilitar la operacion
    x = np.transpose(x)
    # Vector con etiquetas
    y = x[len(x)-1]

    # Borramos el vector con etiquetas de los datos
    x = np.delete(x, len(x)-1, 0)

    # Normalizamos los datos
    x = norm_data(x)
    # X esta normalizada
    # Y contiene un vector con caracteristicas

    return (x, y)


print(load_data()[1])


# save ANN's weights in numpy format and cost in csv format
def save_w(w1, w2, cost):
    # complete

    # load AN's weight in numpy format
    return NotImplemented


def load_w():
    # complete
    return(w1, w2)
#
