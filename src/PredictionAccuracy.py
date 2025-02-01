import numpy as np 
from numba import jit, prange
from numpy.random import default_rng

import time

@jit(nopython=True)
def sampling(prob, L, K, theta):
    # prob: a list of llm success probabilities
    # L: number of LLMs
    # K: number of classes
    # theta: MC round number
    totalPA = 0.0
    IniProb = 1.0
    for l in range(L):
        IniProb *= ((1.0-prob[l])/(K-1.0))
    likeli = np.array([IniProb]*K)  
    for t in range(theta):                
        for l in range(L):
            cid = np.random.uniform(0.0, 1.0)
            cid = int(np.ceil((cid>prob[l])*(cid-prob[l])/((1.0-prob[l])/(K-1.0))))
            assert cid < K
            likeli[cid]*=(prob[l]/(1.0-prob[l])*(K-1.0))                        
        # Deterministic version
        likeli[likeli==1.0]=0.0
        if np.argmax(likeli)==0:        
            totalPA += (1.0/(np.sum(likeli==likeli[0])))
        likeli.fill(IniProb)  
    return totalPA

@jit(nopython=True)
def sampling_backup(prob, L, K, theta):
    newprob = np.array([(K-1.0)/(1.0-x) for x in prob])
    # prob: a list of llm success probabilities
    # L: number of LLMs
    # K: number of classes
    # theta: MC round number
    totalPA = 0.0
    IniProb = 1.0
    for l in range(L):
        IniProb *= ((1.0-prob[l])/(K-1.0))
    #likeli = np.array([IniProb]*K)
    likeliall = np.array([[IniProb]*K]*theta)
    cidall = np.random.rand(theta*L) 
    for t in range(theta):  # if prange for parallel, then we cannot share one likeli array
        likeli = likeliall[t]
        for l in range(L):
            cid = cidall[t*L+l]
            cid = max(int(np.ceil((cid-prob[l])*newprob[l])),0)
            #assert cid < K
            likeli[cid]*=(prob[l]*newprob[l])                        
        # Deterministic version
        likeli[likeli==1.0]=0.0  
        totalPA += (np.argmax(likeli)==0)*(1.0/((likeli==likeli[0]).sum()))
        '''
        if np.argmax(likeli)==0:        
            totalPA += (1.0/((likeli==likeli[0]).sum()))
        '''    
        likeli.fill(IniProb)  
    return totalPA

def ApproPA(prob, K, theta):
    # approximates the accuracy of the combination
    L = len(prob)    
    prob.sort()
    prob = np.array(prob)
    totalPA = sampling(prob, L, K, theta)
    return totalPA/theta

def ExactPA(prob, K):
    L = len(prob)
    totalPA = 0.0
    prob.sort()
    IniProb = 1.0
    for l in range(L):
        IniProb *= ((1.0-prob[l])/(K-1.0))
    likeli = np.array([IniProb]*K)
    obs = np.power(K,L)      
    for idx in range(obs):    
        for l in range(L):
            likeli[idx%K]*=(prob[l]/(1.0-prob[l])*(K-1.0))            
            idx=int(idx/K)        
        # Deterministic version
        if np.argmax(likeli)==0:              
            totalPA += (likeli[0]/(np.sum(likeli==likeli[0])*1.0))
        likeli.fill(IniProb) 
    return totalPA          
