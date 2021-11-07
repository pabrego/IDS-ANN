# Training of ANN using PSO+Pinverse

import pandas     as pd
import numpy      as np
import my_utility as ut

def train_pso(x,y,param):
    """
    Parameters:
        x: Datos
        y: Vector de salida
        param: Configuracion
            param[0]: N° Iteraciones
            param[1]: Numero de nodos ocultos
            param[2]: Numero de particulas
            param[3]: Penalizacion C Pinv.
    """
    X = ut.iniSwarm(param[1],param[2],x.shape[0])
    P = {}
    P['Pos']   = np.zeros(X.shape)               #Best particle position  
    P['Fit']   = np.ones((1,X.shape[0]))*np.inf  #Best particle fitness
    P['gBest'] = np.zeros((1,X.shape[1]))        #Best global solution
    P['gFit']  = np.inf                          #Best global Fitness
    V          = np.zeros(X.shape)               #Velicity  Initial        
    w2Best= np.zeros((1,param[1]))               #Best  output weight     
    Cost  = []
    for iTer in range(param[0]):
        P        = ut.mlp_Fitness(x,y,X,param[3])
        print(P)
        (...)    = ut.upd_pFitness(...)        
        V        = ut.upd_veloc(iTer, X, V)          
        X        = X+V       
    return(w1,w2Best,Cost)

# Beginning ...
def main():
    param       = ut.load_config()    
    x,y         = ut.load_data('train.csv')        
    w1,w2, cost = train_pso(x,y,param)         
    ut.save_w(w1,w2,cost)
       
if __name__ == '__main__':   
	 main()
