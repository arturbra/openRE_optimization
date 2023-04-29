import numpy as np
from Richards_model import run_RE

def run_Richards(prec_input_file, pars):
    # Driving data:
    # Target parameters:
    # pars['thetaR'] = 0.131
    # pars['thetaS'] = 0.396
    # pars['alpha'] = 0.423
    # pars['n'] = 2.06
    # pars['Ks'] = 0.0496
    # pars['neta'] = 0.5 #given
    # pars['Ss'] = 0.000001 #given

    qT = np.loadtxt(prec_input_file, skiprows=1, delimiter=',', usecols=1) / 1000
    qB = np.zeros(len(qT))
    t = np.arange(len(qT))
    dt = t[1] - t[0]

    # Soil properties:
    pars['m'] = 1 - 1 / pars['n']

    # Spatial grid:
    dz = 0.1
    zN = 1.5
    z = np.arange(dz / 2, zN, dz)
    n = len(z)

    # Initial condition:
    psi0 = np.zeros(n) - pars['psi0']

    # Run model m times
    for repeat in range(2):
        psi, wb = run_RE(dt, t, dz, zN, n, psi0, qT, qB, pars)

    return wb
