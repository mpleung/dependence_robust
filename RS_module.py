import numpy as np, math, itertools
from scipy.stats import norm
from scipy.special import binom

def RS_mean_stat(X, R, mu):
    """
    now with b=1

    Outputs M-type statistic squared.
    """
    return ( (np.random.choice(X-mu, R).sum()) / X.std() / math.sqrt(R) ) ** 2 

def XProducts(X, mu):
    X2 = (X - mu) / X.std() 
    allPairs = np.array(list(itertools.combinations(X2, 2))) # array of all pairs of Xs
    return allPairs[:,0] * allPairs[:,1]

def RS_U_stat(XProd, R, mu, n):
    """
    XProd = output of XProducts().
    Outputs U-type statistic. Only works for one-dimensional data.
    """
    return (np.random.choice(XProd, R).sum()) / math.sqrt(R) 

def RS_U_stat_large_sample(X, R, mu):
    """
    Slower version of RS_U_stat that avoids having to use XProducts.
    """
    n = X.shape[0]
    X2 = (X - mu) / X.std() 
    U = 0
    for r in range(R):
        NewPair = np.random.choice(X2,2,replace=False) 
        U += NewPair[0] * NewPair[1]
    return U / math.sqrt(R) - math.sqrt(R) / float(n)

def RS_perm_CV(X, XProd, R, L, alpha, quad):
    """
    Outputs permutation critical value. Only works for one-dimensional data.

    XProd = output of XProducts().
    L = number of draws to obtain permutation CVs.
    alpha = significance level.
    quad = True if you want the U-type statistic, False if you want the M-type statistic.
    """
    Xbar = X.mean()
    PermDist = np.zeros(L)
    if quad:
        for l in range(L):
            PermDist[l] = RS_U_stat(XProd, R, Xbar, X.shape[0])
    else:
        for l in range(L):
            PermDist[l] = RS_mean_stat(X, R, Xbar)

    return np.percentile(PermDist, (1-alpha)*100)

