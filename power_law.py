import numpy as np, pandas as pd, powerlaw as pl, sys, os
from RS_module import *

np.random.seed(seed=0)
np.seterr(divide='ignore', invalid='ignore')

#########

NullDist = 'exponential' # null distribution
DGPs = ['exponential','power_law']

Ns = [100, 500, 1000] # sample size
perm = False # set to true for permutation CVs
L = 1000 # number of permutation draws for calculating permutation CVs
B = 6000 # number of simulations for calculating type I error
alpha = 0.05 # significance level
xmin = 1 # lower support point

LLRatio = np.zeros((len(DGPs),len(Ns))) # log likelihood ratio
pick_ER = np.zeros((len(DGPs),len(Ns))) # fraction of time test picks exponential
pick_PA = np.zeros((len(DGPs),len(Ns)))
Rs = np.zeros(len(Ns))

for cell,DGP in enumerate(DGPs):
    print("\nDGP: " + DGP)

    for i,n in enumerate(Ns):
        print("n = {}".format(n))
        R = int((n/2)**(4/3))
        Rs[i] = R

        vpick_ER = np.zeros(B)
        vpick_PA = np.zeros(B)
        vLLRatio = np.zeros(B)

        for b in range(B):
            ### generate degree distribution ###
            if DGP == 'exponential':
                theoretical_distribution = pl.Exponential(xmin=xmin, parameters=[0.5])
                simulated_data = theoretical_distribution.generate_random(n)
            else:
                theoretical_distribution = pl.Power_Law(xmin=xmin, parameters=[2])
                simulated_data = theoretical_distribution.generate_random(n)

            ### compute MLE and log likelihood for power law ###
            results = pl.Fit(simulated_data, xmin=xmin)
            ParamsPower = results.power_law.alpha

            LogLikFunc = results.power_law.loglikelihoods
            LogLikPower = LogLikFunc(simulated_data)

            ### compute MLE and log likelihood for null distribution ###
            ParamsNull = results.exponential.Lambda
            LogLikFunc = results.exponential.loglikelihoods
            LogLikNull = LogLikFunc(simulated_data)

            ### test ###
            LogLiks = LogLikPower - LogLikNull
            LL = LogLiks.sum() / math.sqrt(LogLiks.shape[0]) / LogLiks.std()
            LLProd = XProducts(LogLiks, 0)
            stat = RS_U_stat(LLProd, R, 0, n)
            LLProdBar = XProducts(LogLiks, LogLiks.mean())
            CV = RS_perm_CV(LogLiks, LLProdBar, R, L, alpha, True)
            vpick_ER[b] = ( stat > CV ) * ( LL < 0 )
            vpick_PA[b] = ( stat > CV ) * ( LL > 0 )
            vLLRatio[b] = LL

        pick_ER[cell,i] = vpick_ER.mean()*100
        pick_PA[cell,i] = vpick_PA.mean()*100
        LLRatio[cell,i] = vLLRatio.mean()

table = pd.DataFrame(np.hstack([ np.vstack([pick_ER[cell,:], pick_PA[cell,:], LLRatio[cell,:], Rs]) for cell in range(len(DGPs)) ]))
table.index = ['Favor Exp', 'Favor PL', 'LL', '$R_n$']
table.columns = pd.MultiIndex.from_product([['Exponential', 'Power Law'], Ns])
print('\n\n\n\\begin{table}[ht]')
print('\centering')
print('\caption{Power Law Test}')
print('\\begin{threeparttable}')
print(table.to_latex(float_format = lambda x: '%.2f' % x, header=True, escape=False, multicolumn_format='c'))
print('\\begin{tablenotes}[para,flushleft]')
print("  \\footnotesize Averages over {} simulations. ``LL'' $=$ average normalized log-likelihood ratio. ``Favor Exp'' $=$ \% rejections in favor of exponential.".format(B))
print('\end{tablenotes}')
print('\end{threeparttable}')
print('\end{table}')
table.to_csv('results_power_law.csv', float_format='%.6f')

