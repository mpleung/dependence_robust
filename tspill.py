import numpy as np, snap, networkx as nx, os, pandas as pd
from scipy.stats import norm, chi2
from RS_module import *
from gen_net_module import *

np.random.seed(seed=0)

def DTG_array(G, D):
    """
    Returns nx3 matrix, with row i corresponding to the treatment status,
    number of treated neighbors, and number of neighbors of node i.

    G = networkx graph object with nodes labeled 0, ..., n-1.
    D = n-dimensional treatment assignment vector.
    """
    num_nhbrs = np.array([G.degree[i] for i in G.nodes])
    num_nhbrs_treated = [D[[j for j in G.neighbors(i)]].sum() for i in G.nodes]
    return np.vstack([D, num_nhbrs_treated, num_nhbrs]).T.astype('float')

### Key user parameters ###

simulate_only = True # if true, simulate expected statistics. after that, then rerun setting this to false to run main simulations
perm = False # set to True to use permutation CVs
alpha = 0.05 # significance level
B = 6000 # number of simulations for calculating type I error / power
L = 1000 # numer of permutation draws for calculating permutation CVs
R_perturb = [0.6,0.8,1,1.2,1.4] # factors to multiply our recommended value of R_n by (to assess robustness)
NumNodes = [100, 500, 1000] # number of observations

### Output ###

if not os.path.exists('output_TS'):
     os.makedirs('output_TS')

### DGP parameters ###

p_d = 0.3 # probability of being treated
theta_nf = np.array([-1, 0.25, 0.25, 1]) # parameters for network model
d = 2
kappa = gen_kappa(theta_nf, d)
theta_linear = np.array([1, 0.5, -1, 0.5])

### Main ###

results_mtype = np.zeros((len(NumNodes),3,len(R_perturb)))
results_utype = np.zeros((len(NumNodes),3,len(R_perturb)))

for cell,n in enumerate(NumNodes):
    print("\n\nn = {}\n".format(n))
    r = (kappa/n)**(1/d)
    W = np.zeros((B,n))

    if simulate_only:
        for b in range(B):
            ### DGP for network ###
            G_temp = gen_SNF(theta_nf, d, n, r, True)
            G = snap_to_nx(G_temp)
            G_sparse = nx.to_scipy_sparse_matrix(G)
            degrees = G_sparse*np.ones(n)

            ### DGP for outcomes ###
            degrees = G_sparse*np.ones(n)
            nu = np.random.normal(0,1,n) 
            eps = G_sparse*nu / (degrees + np.ones(n)*(degrees==0)) + nu # unobservables in linear outcome equation
            D = np.random.binomial(1, p_d, n)
            X = np.hstack([np.ones(n)[:,np.newaxis], DTG_array(G, D)])
            Y = np.dot(X, theta_linear) + eps
            P = np.linalg.inv(X.T.dot(X)/n).dot(X.T)
            W[b,:] = P[2,:] * Y # for inference on spillover parameter

        pd.DataFrame(W).to_csv('output_TS/W_' + str(n) + '.csv', header=False, index=False)

    else:
        W_data = pd.read_csv('output_TS/W_' + str(n) + '.csv', header=None).values
        
        for MType in [True,False]:
            if MType:
                print("Mean-type")
                q = chi2.ppf(1-alpha,1)
                R = int(math.sqrt(n))
            else:
                print("U-type")
                q = norm.ppf(1-alpha)
                R = int((n/2)**(4/3))

            print("    R = {}".format(R))

            for H in ['null','alt']:
                print("    hypothesis: " + H)
                null_spillover = theta_linear[2] if H == 'null' else theta_linear[2]-0.8

                reject = np.zeros((B,len(R_perturb)))

                for b in range(B):
                    W = W_data[b,:]
                    WProd = XProducts(W, null_spillover) if not MType else 0
                    WProdBar = XProducts(W, W.mean()) if perm and not MType else 0

                    for i,perturb in enumerate(R_perturb):
                        R_n = int(np.round(R*perturb))
                        T = RS_U_stat(WProd, R_n, null_spillover, n) if not MType else RS_mean_stat(W, R_n, null_spillover)
                        if perm:
                            reject[b,i] = T > RS_perm_CV(W, WProdBar, R_n, L, alpha, not MType)
                        else:
                            reject[b,i] = T > q

                if MType == True:
                    if H == 'null':
                        results_mtype[cell,0,:] = reject.mean(axis=0)*100
                    else:
                        results_mtype[cell,1,:] = reject.mean(axis=0)*100
                    results_mtype[cell,2,:] = np.round(R*np.array(R_perturb))
                else:
                    if H == 'null':
                        results_utype[cell,0,:] = reject.mean(axis=0)*100
                    else:
                        results_utype[cell,1,:] = reject.mean(axis=0)*100
                    results_utype[cell,2,:] = np.round(R*np.array(R_perturb))
                    
### Output ###

if not simulate_only:
    table = pd.DataFrame(np.vstack( [np.hstack([results_mtype[cell,:,:] for cell in range(len(NumNodes))]), np.hstack([results_utype[cell,:,:] for cell in range(len(NumNodes))])]))
    table.index = pd.MultiIndex.from_product([['M', 'U'], ['Size', 'Power', '$R_n$']])
    table.columns = pd.MultiIndex.from_product([[100, 500, '1k'], R_perturb])
    print('\n\n\n\\begin{table}[ht]')
    print('\centering')
    print('\caption{Treatment Spillovers}')
    print('\\begin{threeparttable}')
    print(table.to_latex(float_format = lambda x: '%.2f' % x, header=True, escape=False, multicolumn_format='c'))
    print('\\begin{tablenotes}[para,flushleft]')
    print("  \\footnotesize Averages over {} simulations. $M =$ mean-type statistic, $U =$ U-type statistic.".format(B))
    print('\end{tablenotes}')
    print('\end{threeparttable}')
    print('\end{table}')
    table.to_csv('results_tspill.csv', float_format='%.6f')

