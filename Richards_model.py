import numpy as np
import pandas as pd
import time
from scipy.integrate import ode, solve_ivp
from numba import jit
from numba import types
from numba.typed import Dict


def make_dict_float():
    d = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float64, )
    return d

@jit(nopython=True)
def theta_fun(psi, pars):
    Se = (1 + (psi * -pars['alpha']) ** pars['n']) ** (-pars['m'])
    Se[psi > 0.] = 1.0
    return pars['thetaR'] + (pars['thetaS'] - pars['thetaR']) * Se


@jit(nopython=True)
def c_fun(psi, pars):
    Se = (1 + (psi * -pars['alpha']) ** pars['n']) ** (-pars['m'])
    Se[psi > 0.] = 1.0
    dSedh = pars['alpha'] * pars['m'] / (1 - pars['m']) * Se ** (1 / pars['m']) * (1 - Se ** (1 / pars['m'])) ** pars[
        'm']
    return Se * pars['Ss'] + (pars['thetaS'] - pars['thetaR']) * dSedh


@jit(nopython=True)
def k_fun(psi, pars):
    Se = (1 + (psi * -pars['alpha']) ** pars['n']) ** (-pars['m'])
    Se[psi > 0.] = 1.0
    return pars['Ks'] * Se ** pars['neta'] * (1 - (1 - Se ** (1 / pars['m'])) ** pars['m']) ** 2


@jit(nopython=True)
def c_inv_fun(psi, psi_n, pars):
    Cinv = 1 / c_fun(psi, pars)
    return Cinv


@jit(nopython=True)
def boundary_fluxes(BC_T, BC_B, pars, dz, psiTn, psiBn):
    # Inputs:
    #  BC_T = specified flux at surface or specified pressure head at surface;
    #  BC_B = specified flux at base or specified pressure head at base;
    # For free drainage BC_B must be an arbitrary array, that is not used.
    #  pars = soil hydraulic properties
    # psiTn = pressure head at node 0 (uppermost node)
    # psiBn = pressure head at node -1 (lowermost node)

    # Upper BC: Type 2 specified flux
    qT = BC_T

    # Lower BC: Free drainage
    qB = k_fun(np.array([psiBn]), pars)[0]
    return qT, qB


# 06_richardsFlux.py
# Functions called by the ODE solver:
def odefun_blockcentered(t, DV, pars, n, BC_T, BC_B, dz, psi_n):
    return odefuncall(t, DV, pars, n, BC_T, BC_B, dz, psi_n)


@jit(nopython=True)
def odefuncall(t, DV, pars, n, BC_T, BC_B, dz, psi_n):
    # In this function, we use a block centered grid approch, where the finite difference
    # solution is defined in terms of differences in fluxes.

    # Unpack the dependent variable:
    QT = DV[0]
    QB = DV[-1]
    psi = DV[1:-1]

    # qT=np.interp(t,tT,qT)
    q = np.zeros(n + 1)
    K = np.zeros(n + 1)

    K = k_fun(psi, pars)
    Kmid = (K[1:] + K[:-1]) / 2.

    # Boundary fluxes:
    qT, qB = boundary_fluxes(BC_T, BC_B, pars, dz, psi[0], psi[-1])
    q[0] = qT
    q[-1] = qB

    # Internal nodes
    q[1:-1] = -Kmid * ((psi[1:] - psi[:-1]) / dz - 1)

    # Continuity
    Cinv = c_inv_fun(psi, psi_n, pars)
    dpsidt = -Cinv * (q[1:] - q[:-1]) / dz

    #    # Change in cumulative fluxes:
    #    dQTdt=qT
    #    dQBdt=qB

    # Pack up dependent variable:
    dDVdt = np.hstack((np.array([qT]), dpsidt, np.array([qB])))

    return dDVdt


# 08_solve_ode_RF_BDF.py
def run_RE(dt, t, dz, zN, n, psi0, BC_T, BC_B, parsIN):
    # 4. scipy function "ode", with the jacobian, solving one step at a time:

    pars = make_dict_float()
    for k in parsIN: pars[k] = parsIN[k]

    DV = np.zeros((len(t), n + 2))
    DV[0, 0] = 0.  # Cumulative inflow
    DV[0, -1] = 0.  # Cumulative outflow
    DV[0, 1:-1] = psi0  # Matric potential

    r = ode(odefun_blockcentered)
    r.set_integrator('vode', method='BDF', uband=1, lband=1, atol=1e-7, rtol=1e-7, max_step=1e10)

    for i, ti in enumerate(t[:-1]):
        r.set_initial_value(DV[i, :], 0)

        params = (pars, n, BC_T[i], BC_B[i], dz, DV[i, :])
        # r.set_jac_params(*params)
        r.set_f_params(*params)
        r.integrate(dt)
        DV[i + 1, :] = r.y

    # Unpack output:
    QT = DV[:, 0]
    QB = DV[:, -1]
    psi = DV[:, 1:-1]
    qT = np.hstack([0, np.diff(QT)]) / dt
    qB = np.hstack([0, np.diff(QB)]) / dt

    # Water balance terms
    theta = theta_fun(psi.reshape(-1), pars)
    theta = np.reshape(theta, psi.shape)
    S = np.sum(theta * dz, 1)

    # Pack output into a dataframe:
    WB = pd.DataFrame(index=t)
    WB['S'] = S
    WB['QIN'] = qT
    WB['QOUT'] = qB

    return psi, WB
