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
def mlp_Fitness(x,y,X,C):
    """
    Parameters: 
        d: dimensión. Numero de variables
        x: Datos de entrenamiento
        y: Valores de salida
        X: Matriz con los pesos
        C: Valor de penalizacion para la pseudo-inversa
    """

    """
    Pasos:
    La neurona i recibe la suma(de j=0 a d) de x[j] * w1[j][i]

    """
    H = np.zeros(X.shape[0])              # matriz de salida de las neuronas
    for i in range(x.shape[0]):         #por cada neurona
        sum                             #sumo todos los valores de entrada (x * w1: revisar lucidchard)
        salida = act_function()         #aplico funcion de activación
        H.append(salida)                #lo agrego a la matriz

        W2 = w2_pinv(H, y, C)           # actualizo valores de W2 con la p-inversa

        MSE                             # calculo error cuadrático

    
    return(W2, MSE)


# Update Particle fitness
def upd_pFitness():
    # complte
    return(...)


# Update Velocity of PSO
def upd_veloc(iTer, X, V):
    # Constantes
    c1 = 1.05
    c2 = 2.95
    r1 = np.random.randint(0, 1)
    r2 = np.random.randint(0, 1)
    # Leer datos de configuración
    params = load_config()
    # Inercia
    '''Imax: iteracion maxima; Iact: iteracion actual'''
    for i in range(len(X)):
        Imax = params[0]
        a = 0.95 -( ((0.95 - 0.1) / Imax) * iTer)
        # Velocidad (V)
        V = a * 

    return(...)


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
def forward():
    # complete
    return(a)


def act_function(w, X):
    # complete h(w, X)
    h = np.exp(w * X) - np.exp((-1 * w) * X )
    h /= np.exp(w * X) + np.exp((-1 * w) * X )
    
    return(h)

# Measure
def metrica(y, z):
    # complete
    return(...)


# Confusion matrix
def confusion_matrix():
    # complete
    return(...)


# Norm of the data
def norm_data(x):
    """
    Esta funcion se encarga de normalizar los datos.

    Returns: 
    xn: Parametro x normalizado

    TODO: Hablar con el grupo para ver si X deberia contener a Y, dado que estos valores se separan

    """
    # Constantes
    a = 0.01
    b = 0.99

    # Se transpone la matriz para una facilitar la operacion

    ''' Tener cuidado con el número de iteraciones al recorrer la matriz transpuesta, no se debe considerar el valor Y '''
    xn = np.transpose(x)

    # Es mas facil normalizar todo el vector, y reemplazar el ultimo elemento
    for i in range(len(xn)):
        # Normalizar todo el vector
        col = xn[i]
        # Proceso de normalizacion
        aux = np.divide((col - np.amin(col)), (np.amax(col)-np.amin(col)))
        col = aux*(b-a)+a

        # Se reemplaza la fila original con la normalizada
        xn[i] = col

    # Dehacemos la transposicion
    xn = np.transpose(xn)
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
def load_data(fname):

    fname = 'D:\Escritorio\Codigo\Python\TareaIDS\Datos\\train.csv'
    # Constantes
    x = pd.read_csv(fname, header=None)
    x = np.array(x)
    # Vector con caracteristicas
    y = x[len(x)-1]

    x = norm_data(x)

    # X esta normalizada
    # Y contiene un vector con caracteristicas

    # TODO: Revisar si X deberia contener Y

    return (x, y)



# save ANN's weights in numpy format and cost in csv format
def save_w(w1, w2, cost):
    # complete

    # load AN's weight in numpy format
    return NotImplemented


def load_w():
    # complete
    return(w1, w2)
#
