import numpy as np, pandas as pd, scipy.sparse, math
from scipy.stats import norm, chi2
from RS_module import *

np.random.seed(seed=0)

### Key user parameters ###

perm = False # set to True to use permutation CVs
alpha = 0.05 # significance level
B = 6000 # number of simulations for calculating type I error / power
L = 1000 # number of permutation draws for calculating permutation CVs
R_perturb = [0.6,0.8,1,1.2,1.4] # factors to multiply our recommended value of R_n by (to assess robustness)

# cluster sizes
Ns = [ [20, 100, 200], [20, 500, 1000] ]

true_beta = 1.0 

### Main ###

# small/large refers to sample size
results_mtype_small = np.zeros((3,len(R_perturb))) # rows are size / power / R_n, columns are R_perturb
results_mtype_large = np.zeros((3,len(R_perturb)))
results_utype_small = np.zeros((3,len(R_perturb)))
results_utype_large = np.zeros((3,len(R_perturb)))
results_ttest_small = np.zeros((2,len(Ns[0]))) # rows are size / power, columns are cluster level
results_ttest_large = np.zeros((2,len(Ns[0])))

for cell,num_clusters in enumerate(Ns):
    n = num_clusters[2] # number of observations
    A = [scipy.sparse.kron( np.identity(num), np.ones((int(np.round(n/num)), \
            int(np.round(n/num)))), 'csr' ) for num in num_clusters] # clustering structure for SEs
    print("\n\nn = {}\n".format(n))

    for MType in [True,False]:
        if MType:
            print("Mean-type")
            q = chi2.ppf(1-alpha,1)
            R = int(np.round(math.sqrt(n)))
        else:
            print("U-type")
            q = norm.ppf(1-alpha)
            R = int(np.round((n/2)**(4/3)))

        print("    R = {}".format(R))

        for H in ['null','alt']:
            print("    hypothesis: " + H)
            null_beta = 1.0 if H == 'null' else 1.5

            T = np.zeros((B,len(R_perturb))) # RS test stats 
            CV = q * np.ones((B,len(R_perturb))) # critical value
            ttest_reject = np.zeros((B,len(num_clusters))) # clustered t-test rejection indicators

            for i,perturb in enumerate(R_perturb):
                R_n = int(np.round(R*perturb))

                for b in range(B):
                    eps = np.random.normal(0,1,n)
                    RandomEffect = np.kron( np.random.normal(0,1,num_clusters[1]), \
                            np.ones(int(np.round(n/num_clusters[1]))) ) # family-level random shock
                    Y = true_beta + RandomEffect + eps
                    YProd = XProducts(Y,null_beta) if not MType else 0
                    YProdBar = XProducts(Y,Y.mean()) if perm and not MType else 0

                    # Clustered SEs
                    if MType: 
                        est = Y.mean()
                        resid = Y - est
                        for j,num in enumerate(num_clusters):
                            SE = np.sqrt( resid.T.dot(scipy.sparse.csr_matrix.dot(A[j],resid)) ) / n
                            ttest_reject[b,j] = abs((est - null_beta) / SE) > norm.ppf(1-alpha/2)
                        
                    # Our method
                    T[b,i] = RS_U_stat(YProd, R_n, null_beta, n) if not MType else RS_mean_stat(Y, R_n, null_beta)
                    if perm: CV[b,i] = RS_perm_CV(Y, YProdBar, R_n, L, alpha, not MType) 
            
            myTypeI = np.around((T > CV).mean(axis=0)*100, 2)
            if MType: tTypeI = np.around(ttest_reject.mean(axis=0)*100, 2)

            if cell == 0:
                if MType == True:
                    if H == 'null':
                        results_mtype_small[0,:] = myTypeI
                        results_ttest_small[0,:] = tTypeI
                    else:
                        results_mtype_small[1,:] = myTypeI
                        results_ttest_small[1,:] = tTypeI
                    results_mtype_small[2,:] = np.round(R*np.array(R_perturb))
                else:
                    if H == 'null':
                        results_utype_small[0,:] = myTypeI
                    else:
                        results_utype_small[1,:] = myTypeI
                    results_utype_small[2,:] = np.round(R*np.array(R_perturb))
            else:
                if MType == True:
                    if H == 'null':
                        results_mtype_large[0,:] = myTypeI
                        results_ttest_large[0,:] = tTypeI
                    else:
                        results_mtype_large[1,:] = myTypeI
                        results_ttest_large[1,:] = tTypeI
                    results_mtype_large[2,:] = np.round(R*np.array(R_perturb))
                else:
                    if H == 'null':
                        results_utype_large[0,:] = myTypeI
                    else:
                        results_utype_large[1,:] = myTypeI
                    results_utype_large[2,:] = np.round(R*np.array(R_perturb))

table1 = pd.DataFrame(np.vstack([ np.hstack([results_mtype_small, results_mtype_large]), np.hstack([results_utype_small, results_utype_large]) ]))
table1.index = pd.MultiIndex.from_product([['M', 'U'], ['Size', 'Power', '$R_n$']])
table1.columns = pd.MultiIndex.from_product([['$(20,100,200)$', '$(20,500,1000)$'], R_perturb])
print('\n\n\n\\begin{table}[ht]')
print('\centering')
print('\caption{Cluster Dependence: Our Tests}')
print('\\begin{threeparttable}')
print(table1.to_latex(float_format = lambda x: '%.2f' % x, header=True, escape=False, multicolumn_format='c'))
print('\\begin{tablenotes}[para,flushleft]')
print("  \\footnotesize Averages over {} simulations. $N = (n_c,n_f,n_i)$. $M =$ mean-type test, $U =$ U-type test.".format(B,int((1-alpha)*100)))
print('\end{tablenotes}')
print('\end{threeparttable}')
print('\end{table}')

table2 = pd.DataFrame(np.hstack([results_ttest_small, results_ttest_large]))
table2.index = ['Size', 'Power']
table2.columns = pd.MultiIndex.from_product([['$(20,100,200)$', '$(20,500,1000)$'], ['$c$', '$f$', '$i$', '$c$', '$f$', '$i$']])
print('\n\n\n\\begin{table}[ht]')
print('\centering')
print('\caption{Cluster Dependence: $t$-Tests}')
print('\\begin{threeparttable}')
print(table2.to_latex(float_format = lambda x: '%.2f' % x, header=True, escape=False, multicolumn_format='c'))
print('\\begin{tablenotes}[para,flushleft]')
print("  \\footnotesize Averages over {} simulations. $N = (n_c,n_f,n_i)$.".format(B,int((1-alpha)*100)))
print('\end{tablenotes}')
print('\end{threeparttable}')
print('\end{table}')

table1.to_csv('results_cluster_mytest.csv', float_format='%.6f')
table2.to_csv('results_cluster_ttest.csv', float_format='%.6f')

