import numpy as np
import pandas as pd

from Richards_model import run_RE


def run_Richards(wb, pars):
    # Driving data:
    qT = pd.read_csv(wb)['rain'] * 0.0254
    qB = np.zeros(len(qT))
    t = np.arange(len(qT))
    dt = 5

    pars['m'] = 1 - 1 / pars['n']

    # Spatial grid:
    dz = 0.1
    zN = 0.8
    z = np.arange(dz / 2, zN, dz)
    n = len(z)

    # Initial condition:
    psi0 = np.zeros(n) - 10

    # Run model m times
    psi, wb = run_RE(dt, t, dz, zN, n, psi0, qT, qB, pars)

    return wb
