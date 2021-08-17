Replication files for "Dependence-Robust Inference Using Resampled Statistics". The files were coded for Python 3 and require the following dependencies: networkx, numpy, pandas, [powerlaw](https://github.com/jeffalstott/powerlaw), scipy, and [snap](https://snap.stanford.edu/snappy/index.html).

Contents:
* RS\_module.py: Functions implementing our methods.
* gen\_net\_module.py: Functions for simulating data.
* jackson\_rogers\_application.py: Empirical application. Prints output as latex table directly to console.
* jackson\_rogers\_data: Data for empirical application. Obtained from http://www.stanford.edu/~jacksonm/JacksonRogers-Data.zip. 
* clustering.py: Clustered data monte carlo (Tables 2 and 3). Prints output as latex table and also saves as CSV in this directory.
* clustering\_strongdep.py: Clustered data with strong dependence monte carlo (results verbally summarized in section 6). Prints output as latex table directly to console and also saves as CSV in this directory.
* node\_stats.py: Network statistics monte carlo (Tables B.1 and B.2). Prints output as latex table directly to console and also saves as CSV in this directory. First run with simulate\_only=True. The rerun with simulate\_only=False to get the results.
* power\_law.py: Power law monte carlo (Table B.4). Prints output as latex table directly to console and also saves as CSV in this directory.
* tspill.py: Treatment spillovers monte carlo (Table B.3). Prints output as latex table directly to console and also saves as CSV in this directory. First run with simulate\_only=True. The rerun with simulate\_only=False to get the results.

