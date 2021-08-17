import numpy as np, pandas as pd, powerlaw as pl, snap
from scipy.stats import norm
from RS_module import *
from gen_net_module import *

np.random.seed(seed=0)
np.seterr(divide='ignore', invalid='ignore')

#########

R_perturb = [0.6,0.8,1,1.2,1.4] # factors to multiply our recommended value of R_n by (to assess robustness)
files = ['coauthor','ham_radio','prison','romance','citations','www']
NullDist = 'exponential' # null distribution
if NullDist not in ['exponential', 'lognormal', 'stretched_exponential', 'gamma']:
    raise ValueError('Not supported.')

alpha = 0.05 # significance level
xmin = 1 # lower support point

results_RS = np.zeros((len(files),len(R_perturb))) # results of my test
Rs = np.zeros((len(files),len(R_perturb))) # values of R_n
results_naive = np.zeros(len(files)) # results of standard Vuong test
ExponentEsts = np.zeros(len(files)) # parameter estimate under power law
LambdaEsts = np.zeros(len(files)) # parameter estimate under exponential
xmins = np.zeros(len(files)) # lower support point
ns = np.zeros(len(files)) # sample size
LLRatios = np.zeros(len(files)) # log-likelihood ratio

for cell,DegData in enumerate(files):
    print(DegData + ".csv")

    ### read in data ###
    data = pd.read_csv('jackson_rogers_data/' + DegData + '.csv', header=None).values
    degrees = np.array([])
    for i in range(data.shape[0]):
        if data[i,1] > 0:
            degrees = np.append(degrees, np.ones(data[i,1])*data[i,0])
    n = degrees.shape[0]

    ### compute MLE and log likelihood for power law ###
    results = pl.Fit(degrees, discrete=True, xmax=None, estimate_discrete=True, xmin=xmin)
    # Note: the warning about throwing out zero values is just because we have to choose xmin > 0 for a power law to be well-defined.
    xmin = results.power_law.xmin
    ParamsPower = results.power_law.alpha
    LogLikFunc = results.power_law.loglikelihoods
    LogLikPower = LogLikFunc(degrees)

    ### compute MLE and log likelihood for null distribution ###
    ParamsNull = results.exponential.Lambda
    LogLikFunc = results.exponential.loglikelihoods
    LogLikNull = LogLikFunc(degrees)

    ### test ###
    R = min(int(np.round((n/2)**(4/3))), 100000)
    LogLiks = LogLikPower - LogLikNull
    LL = LogLiks.sum() / math.sqrt(LogLiks.shape[0]) / LogLiks.std()
    CV = norm.ppf(1-alpha)

    for i,perturb in enumerate(R_perturb):
        R_n = int(np.round(R*perturb))
        Rs[cell,i] = R_n
        Ustat = RS_U_stat_large_sample(LogLiks, R_n, 0)
        if Ustat > CV and LL < 0:
            results_RS[cell,i] = 1 # exponential
        elif Ustat > CV and LL > 0:
            results_RS[cell,i] = 2 # power law

    ns[cell] = n
    if LL < norm.ppf(alpha):
        results_naive[cell] = 1 # exponential
    elif LL > norm.ppf(1-alpha):
        results_naive[cell] = 2 # power law
    ExponentEsts[cell] = ParamsPower
    LambdaEsts[cell] = ParamsNull
    xmins[cell] = xmin
    ns[cell] = degrees[degrees>=xmin].shape[0]
    LLRatios[cell] = LL

### output ###

JRr = np.array([4.7, 5.0, -1, -1, 0.63, 0.57]) 

table = pd.DataFrame(np.vstack([ExponentEsts, LLRatios, results_naive, JRr, ns]))
table.index = ['Exp.\\', 'LL', 'Naive', '$r$', '$n$']
table.columns = ['Coauthor','Radio','Prison','Romance','Citation','WWW']
table.replace([-1,0,1,2],['$\\infty$','N','E','P'], inplace=True)
print(table.to_latex(float_format = lambda x: '%.2f' % x, header=True, escape=False, multicolumn_format='c'))

table = pd.DataFrame( np.hstack([np.hstack([Rs[cell,:][:,None], results_RS[cell,:][:,None]]) for cell in range(len(files))]) )
table.columns = pd.MultiIndex.from_product([['Coauthor','Radio','Prison','Romance','Citation','WWW'], ['$R_n$','RS']])
table.replace([-1,0,1,2],['$\\infty$','N','E','P'], inplace=True)
print('\\begin{table}[ht]')
print('\centering')
print('\caption{Results}')
print('\\begin{threeparttable}')
print(table.to_latex(float_format = lambda x: '%.2f' % x, header=True, escape=False, multicolumn_format='c'))
print('\\begin{tablenotes}[para,flushleft]')
print("  \\footnotesize ``Exp.'' $=$ estimated power law exponent. ``LL'' $=$ the normalized log-likelihood ratio. ``RS'' $=$ conclusion of our test, and ``Naive'' $=$ conclusion of i.i.d.\ test, where P $=$ power law, E $=$ exponential, and N $=$ fail to reject.")
print('\end{tablenotes}')
print('\end{threeparttable}')
print('\end{table}\n')

