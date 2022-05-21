import numpy as np
from sklearn.neighbors import NearestNeighbors


def discrepancy(embs, case1, case2, scale=1e-3, n_neighbors=3+1):
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(embs)

    _, case1_neighbor = knn.kneighbors(case1)
    case1_neighbor = embs[case1_neighbor][0]
    case1_inter = 1/np.sum(np.exp(scale*(case1@case1_neighbor.T)))

    _, case2_neighbor = knn.kneighbors(case2)
    case2_neighbor = embs[case2_neighbor][0]
    case2_inter = 1/np.sum(np.exp(scale*(case2@case2_neighbor.T)))

    result = np.exp(scale*np.inner(case1, case2))*(case1_inter+case2_inter)
    # print(f"DAM : {result}")
    return result[0][0]


def cosig(case1, case2):
    result = case2 @ case1.T
    result = result/np.sqrt(np.sum(case1**2))/np.sqrt(np.sum(case2**2))
    result = 1/(1+np.exp(-result))
    return result[0][0]
