import numpy as np
from Richards_model import run_RE

def run_Richards(prec_input_file, pars):
    # Driving data:
    # Target parameters:
    # pars = {'thetaR': 0.131, 'thetaS': 0.396, 'alpha': 0.423, 'n': 2.06, 'Ks': 3.32, 'psi0': 13.5, 'neta': 0.5,
    #         'Ss': 0.000001}

    qT = np.loadtxt(prec_input_file, skiprows=1, delimiter=',', usecols=1) / 1000
    qT = qT[:int(len(qT)/4)]
    zeros_begin = np.zeros(10)
    zeros_end = np.zeros(40)
    qT = np.concatenate([zeros_begin, qT])
    qT = np.concatenate([qT, zeros_end])
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
    psi, wb = run_RE(dt, t, dz, zN, n, psi0, qT, qB, pars)

    return wb
