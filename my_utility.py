# My Utility : auxiliars functions
import pandas as pd
import numpy  as np

#Initialize  Swarm
def iniSwarm(Nh,Np,d):
    X = np.zeros((Np,(Nh*d)))
    #complete
    return(X)
# Initialize random weights
def randW(next,prev):
    #complete
    return(w)
# MLP-fitness
def mlp_Fitness():    
    #complete
    return(W2,MSE)

# Update Particle fitness
def upd_pFitness():    
    #complte            
    return(...)    

# Update Velocity of PSO
def upd_veloc():
    #complrte
    return(...)

# Update Swarm of QPSO
def upd_swarm():
    #complete    
    return(...)

# Update W2 via P-inverse 
def w2_pinv(H,y,C):
    #complete    
    return(w2)

# Feed-forward of ANN
def forward():    
    #complete
    return(a)

# activation function
def act_function(...):
    #complete
    return(...)   
 
# Measure
def metrica(y,z):
    #complete
    return(...)
#Confusion matrix
def confusion_matrix():
    #complete
    return(...)

# Norm of the data 
def norm_data(x):
    #complete
    return(xn)
#

#------------------------------------------------------------------------
#      LOAD-SAVE
#-----------------------------------------------------------------------
# Configuration of the ANN
def load_config():      
    param = np.genfromtxt("config_ann.csv",delimiter=',',dtype=None)    
    par=[]    
    par.append(np.int16(param[0])) # Max. Iterations 
    par.append(np.int16(param[1])) # Number of nodes
    par.append(np.int16(param[2])) # Number of particle
    par.append(np.float(param[3])) # Penalidad C of the Pinv.   
    return(par)
# Load data 
def load_data(fname):
    x  = pd.read_csv(fname, header = None)
    x  = np.array(x)
    #complete   
    return(x,y)

# save ANN's weights in numpy format and cost in csv format
def save_w(w1,w2,cost): 
    #complete    

#load AN's weight in numpy format
def load_w():
    #complete
    return(w1,w2)      
#
















