import math
import numpy as np

from MKnapsack import MKnapsack

import time
from PredictionAccuracy import ApproPA

def LLM_call(idx, query, ds):
    # idx: model index
    # ds: ThriftDataset instance, from the test set
    # no need for the real time querying, directly read from the dataset
    q_i = ds.query2idx[query]
    m_i = idx
    return ds.data[q_i, m_i, 1]
    
def AdaptSelect(T, query, prob, K, thrift_ds):
    phi = [1.0] * K
    F_T = 1.0

    tep = []
    S = []
    for i in T:
        sp = prob[i]
        F_T *= (sp * (K-1)/(1.0 - sp))
        tep.append((prob[i],i))
    tep.sort(key=lambda x: x[0])
    T.clear()
    for (a,b) in tep:
        T.append(b)
    while len(T) > 0: 
        H1 = max(phi)
        idx = phi.index(H1)
        phi[idx] = 0.9   # empirical value as placeholder as long as it is smaller than 1.0
        H2 = max(phi)
        phi[idx] = H1
        if F_T * H2 > H1:
            model_idx = T[-1]
            T.remove(model_idx)
            S.append(model_idx)
            sp = prob[model_idx]
            F_T /= (sp * (K-1)/(1.0 - sp))
            q_i = thrift_ds.query2idx[query]
            pre_cls = thrift_ds.data[q_i, model_idx, 1]
            phi[pre_cls] *= (sp * (K-1)/(1.0 - sp))
        else:
            break
    H1 = max(phi)
    pre = phi.index(H1)
    return T + S, S, pre

############ two baseline methods below ###########
def Greedy(prob, cost, B, K, theta):
    L = len(prob)

    ratio = [(prob[i] / cost[i], i) for i in range(L)]
    ratio = sorted(ratio, key=lambda x: x[0], reverse=True)
    
    S = []
    newB = B
    model = [x for x in range(L)]

    for (_,idx) in ratio:
        if newB > cost[idx]:
            S.append(idx)
            newB -= cost[idx]
            model.remove(idx)
        if len(S) == 2: break
    
    if len(S) < 2: return S

    curAcc = max(prob[S[0]], prob[S[1]])

    while True:
        wv = -0.1
        idx = -1
        for x in model:
            if newB > cost[x]:
                newS = S + [x]
                newAcc = ApproPA([prob[i] for i in newS], K, theta)
                ratio = (newAcc - curAcc) / cost[x]
                if ratio > wv:
                    wv = ratio
                    idx = x
        if idx == -1: break
        S.append(idx)
        model.remove(idx)
        newB -= cost[idx]
        curAcc = ApproPA([prob[i] for i in S], K, theta)

    return S

def pstar(prob, cost, B):
    L = len(prob)
    p = 0.0
    idx = -1
    for i in range(L):
        if cost[i] < B:
            if prob[i] > p:
                p = prob[i]
                idx = i
    return idx, prob[idx]

def submodularGreedy(prob, cost, B):
    
    idx,_ = pstar(prob, cost, B)
    S1 = [idx]

    L = len(prob)
    S2 = []
    model = [(prob[i] / cost[i], i) for i in range(L)]
    model = sorted(model, key=lambda x: x[0], reverse=True)
    
    for (_,b) in model:
        if B > cost[b]:
            S2.append(b)
            B -= cost[b]
    return S1, S2

def SanGreedy(prob, cost, B, K, theta):

    S1, S2 = submodularGreedy(prob, cost, B)
    Acc =  ApproPA([prob[i] for i in S2], K, theta)

    if len(S2) <= 2:
        if Acc > prob[S1[0]]: return S2
        else: return S1

    L = len(prob)
    model = [x for x in range(L)]

    S = [S2[0],S2[1]]
    model.remove(S2[0])
    model.remove(S2[1])
    curAcc = max(prob[S2[0]], prob[S2[1]])
    newB = B - cost[S2[0]] - cost[S2[1]]

    while True:
        wv = -0.1
        idx = -1
        for x in model:
            if newB > cost[x]:
                newS = S + [x]
                newAcc = ApproPA([prob[i] for i in newS], K, theta)
                ratio = (newAcc - curAcc) / cost[x]
                if ratio > wv:
                    wv = ratio
                    idx = x
        if idx == -1: break
        S.append(idx)
        model.remove(idx)
        newB -= cost[idx]
        curAcc = ApproPA([prob[i] for i in S], K, theta)
    
    
    TotalAcc = [prob[S1[0]], Acc, curAcc]
    TotalSet = [S1, S2, S]
    idx = np.argmax(TotalAcc)
    return TotalSet[idx]

def Octopus(prob, cost, B):
    L = prob.size
    S = []
    ######### select the first model ########
    idx = -1
    p = -1.0
    for i in range(L):
        if B >= cost[i] and p < prob[i]:
            idx = i
            p = prob[i]
    if idx == -1:
        return S        
    S.append(idx)
    return S

def Prediction(S, prob, query, thrift_ds, K):
    phi = [1.0] * K
    for model_idx in S:
        sp = prob[model_idx]
        pre_cls = LLM_call(model_idx, query, thrift_ds)
        phi[pre_cls] *= (sp * (K-1)/(1.0 - sp))
    H1 = max(phi)
    pre = phi.index(H1)
    return pre


def AdaptSelect_prefix(query, prob, K, thrift_ds, prefix):
    # prefix: a prefixed subset of llms
    phi = [1.0] * K
    F_T = 1.0
    T = prefix
    tep = []
    S = []
    for i in T:
        sp = prob[i]
        F_T *= (sp * (K-1)/(1.0 - sp))
        tep.append((prob[i],i))
    tep.sort(key=lambda x: x[0])
    T.clear()
    for (a,b) in tep: 
        T.append(b)
    while len(T) > 0:
        H1 = max(phi)
        idx = phi.index(H1)
        phi[idx] = 0.9
        H2 = max(phi)
        phi[idx] = H1
        if F_T * H2 > H1:
            model_idx = T[-1]
            S.append(model_idx)
            T.remove(model_idx)
            sp = prob[model_idx]
            F_T /= (sp * (K-1)/(1.0 - sp))
            q_i = thrift_ds.query2idx[query]
            pre_cls = thrift_ds.data[q_i, model_idx, 1]
            phi[pre_cls] *= (sp * (K-1)/(1.0 - sp))
        else:
            break
    H1 = max(phi)
    pre = phi.index(H1)
    return S, pre
