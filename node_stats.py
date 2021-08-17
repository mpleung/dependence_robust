import numpy as np, pandas as pd, snap, math, sys, os
from scipy.stats import norm, chi2
from scipy.special import binom
from gen_net_module import *
from RS_module import *

np.random.seed(seed=0)

### Key user parameters ###

simulate_only = True # if true, this only simulates expected statistics. after that, rerun this, setting this to false to get the final results
model = 'SNF' # options: 'ER' (erdos renyi), 'RGG' (random geometric graph), 'SNF' (strategic network formation)
perm = False # set to True to use permutation CVs
alpha = 0.05 # significance level
B = 6000 # number of simulations for calculating type I error / power
L = 1000 # numer of permutation draws for calculating permutation CVs
R_perturb = [0.6,0.8,1,1.2,1.4] # factors to multiply our recommended value of R_n by (to assess robustness)
NumNodes = [100,500,1000] # number of nodes

### Network DGP ###

theta = np.array([-1, 0.25, 0.25, 1])
d = 2
SNF_kappa = gen_kappa(theta, d)
RGG_kappa = 7/math.pi
edeg = 7

### Output ###

if not os.path.exists('output_NS'):
     os.makedirs('output_NS')

### Main ###

results_mtype_cc = np.zeros((len(NumNodes),3,len(R_perturb)))
results_utype_cc = np.zeros((len(NumNodes),3,len(R_perturb)))
results_mtype_deg = np.zeros((len(NumNodes),3,len(R_perturb)))
results_utype_deg = np.zeros((len(NumNodes),3,len(R_perturb)))

for cell,n in enumerate(NumNodes):
    print("\n\nn = {}\n".format(n))

    ### Simulate expected statistics ###
    if simulate_only:
        # first B draws used for inference, last B draws used to simulate expected statistics
        cc = np.zeros((2*B,n)) # clustering coefficient for each node in each simulation
        degrees = np.zeros((2*B,n)) # degree for each...

        for b in range(B*2): 
            # simulate network
            if model == 'RGG':
                G = gen_RGG(np.random.uniform(0,1,(n,d)), (RGG_kappa/n)**(1/d))
            elif model == 'ER':
                G = snap.GenRndGnm(snap.PUNGraph, n, int(binom(n,2)*edeg/n))
            else:
                G = gen_SNF(theta, d, n, (SNF_kappa/n)**(1/d), True)

            # extract clustering coefficients
            cc_vector = snap.TIntFltH()
            snap.GetNodeClustCf(G, cc_vector)
            cc[b,:] = np.array([cc_vector[i] for i in cc_vector])

            # extract degrees
            deg_vector = snap.TIntPrV()
            snap.GetNodeOutDegV(G, deg_vector)
            degrees[b,:] = np.array([i.GetVal2() for i in deg_vector])

        # save data
        pd.DataFrame(cc).to_csv('output_NS/cc_' + model + '_' + str(n) + '.csv', header=False, index=False)
        pd.DataFrame(degrees).to_csv('output_NS/degrees_' + model + '_' + str(n) + '.csv', header=False, index=False)

    ### Inference ###
    else:
        # load data
        cc_data = pd.read_csv('output_NS/cc_' + model + '_' + str(n) + '.csv', header=None).values
        degrees_data = pd.read_csv('output_NS/degrees_' + model + '_' + str(n) + '.csv', header=None).values

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
                mu_cc = cc_data[B:(2*B),:].mean(axis=1).mean(axis=0)
                mu_deg = degrees_data[B:(2*B),:].mean(axis=1).mean(axis=0)
                if H == 'alt':
                    mu_cc += 0.08
                    mu_deg += 0.8
                    print("cc: {}, deg: {}\n".format(mu_cc, mu_deg))

                reject_cc = np.zeros((B,len(R_perturb)))
                reject_deg = np.zeros((B,len(R_perturb)))

                for b in range(B):
                    cc = cc_data[b,:]
                    degrees = degrees_data[b,:]
                    if MType:
                        ccProd,degProd,ccProdBar,degProdBar = 0,0,0,0
                    else:
                        ccProd = XProducts(cc, mu_cc)
                        degProd = XProducts(degrees, mu_deg)
                        ccProdBar,degProdBar=0,0
                        if perm:
                            ccProdBar = XProducts(cc, cc.mean())
                            degProdBar = XProducts(degrees, degrees.mean())
                    
                    for i,perturb in enumerate(R_perturb):
                        R_n = int(np.round(R*perturb))        
                        T_cc = RS_U_stat(ccProd, R_n, mu_cc, n) if not MType else RS_mean_stat(cc, R_n, mu_cc)
                        T_deg = RS_U_stat(degProd, R_n, mu_deg, n) if not MType else RS_mean_stat(degrees, R_n, mu_deg)
                        if perm:
                            reject_cc[b,i] = T_cc > RS_perm_CV(cc, ccProdBar, R_n, L, alpha, not MType)
                            reject_deg[b,i] = T_deg > RS_perm_CV(degrees, degProdBar, R_n, L, alpha, not MType)
                        else:
                            reject_cc[b,i] = T_cc > q
                            reject_deg[b,i] = T_deg > q

                if MType == True:
                    if H == 'null':
                        results_mtype_cc[cell,0,:] = reject_cc.mean(axis=0)*100
                        results_mtype_deg[cell,0,:] = reject_deg.mean(axis=0)*100
                    else:
                        results_mtype_cc[cell,1,:] = reject_cc.mean(axis=0)*100
                        results_mtype_deg[cell,1,:] = reject_deg.mean(axis=0)*100
                    results_mtype_cc[cell,2,:] = np.round(R*np.array(R_perturb))
                    results_mtype_deg[cell,2,:] = np.round(R*np.array(R_perturb))
                else:
                    if H == 'null':
                        results_utype_cc[cell,0,:] = reject_cc.mean(axis=0)*100
                        results_utype_deg[cell,0,:] = reject_deg.mean(axis=0)*100
                    else:
                        results_utype_cc[cell,1,:] = reject_cc.mean(axis=0)*100
                        results_utype_deg[cell,1,:] = reject_deg.mean(axis=0)*100
                    results_utype_cc[cell,2,:] = np.round(R*np.array(R_perturb))
                    results_utype_deg[cell,2,:] = np.round(R*np.array(R_perturb))

### Output ###

if not simulate_only:
    table = pd.DataFrame( np.vstack([np.hstack([results_mtype_cc[cell,:,:], results_utype_cc[cell,:,:]]) for cell in range(len(NumNodes))]) )
    table.index = pd.MultiIndex.from_product([[100, 500, '1k'], ['Size', 'Power', '$R_n$']])
    table.columns = pd.MultiIndex.from_product([['Mean-Type Test', 'U-Type Test'], R_perturb])
    print('\n\n\n\\begin{table}[ht]')
    print('\centering')
    print('\caption{Average Clustering}')
    print('\\begin{threeparttable}')
    print(table.to_latex(float_format = lambda x: '%.2f' % x, header=True, escape=False, multicolumn_format='c'))
    print('\\begin{tablenotes}[para,flushleft]')
    print("  \\footnotesize Averages over {} simulations. ``Size'' rows obtained from testing $H_0\colon \\theta_0=\\theta^*$, where $\\theta^*=$ true expected value of average clustering. ``Power'' rows obtained from testing $H_0\colon \\theta_0=\\theta^*+0.08$.".format(B))
    print('\end{tablenotes}')
    print('\end{threeparttable}')
    print('\end{table}')
    table.to_csv('results_node_stats_cc.csv', float_format='%.6f')

    table = pd.DataFrame( np.vstack([np.hstack([results_mtype_deg[cell,:,:], results_utype_deg[cell,:,:]]) for cell in range(len(NumNodes))]) )
    table.index = pd.MultiIndex.from_product([[100, 500, '1k'], ['Size', 'Power', '$R_n$']])
    table.columns = pd.MultiIndex.from_product([['Mean-Type Test', 'U-Type Test'], R_perturb])
    print('\n\n\n\\begin{table}[ht]')
    print('\centering')
    print('\caption{Average Degree}')
    print('\\begin{threeparttable}')
    print(table.to_latex(float_format = lambda x: '%.2f' % x, header=True, escape=False, multicolumn_format='c'))
    print('\\begin{tablenotes}[para,flushleft]')
    print("  \\footnotesize Averages over {} simulations. ``Size'' rows obtained from testing $H_0\colon \\theta_0=\\theta^*$, where $\\theta^*=$ true expected value of average degree. ``Power'' rows obtained from testing $H_0\colon \\theta_0=\\theta^*+0.8$.".format(B))
    print('\end{tablenotes}')
    print('\end{threeparttable}')
    print('\end{table}')
    table.to_csv('results_node_stats_deg.csv', float_format='%.6f')

