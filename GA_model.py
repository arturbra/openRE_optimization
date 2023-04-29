import configparser
import run_richards_benchmark
import run_richards_pp
import pandas as pd
import numpy as np


class Calibration:
    def __init__(self, setup_file):
        setup = configparser.ConfigParser()
        setup.read(setup_file)
        self.thetaR_min = float(setup['FLOW_CALIBRATION']['thetaR_min'])
        self.thetaR_max = float(setup['FLOW_CALIBRATION']['thetaR_max'])
        self.thetaS_min = float(setup['FLOW_CALIBRATION']['thetaS_min'])
        self.thetaS_max = float(setup['FLOW_CALIBRATION']['thetaS_max'])
        self.alpha_min = float(setup['FLOW_CALIBRATION']['alpha_min'])
        self.alpha_max = float(setup['FLOW_CALIBRATION']['alpha_max'])
        self.n_min = float(setup['FLOW_CALIBRATION']['n_min'])
        self.n_max = float(setup['FLOW_CALIBRATION']['n_max'])
        self.Ks_min = float(setup['FLOW_CALIBRATION']['Ks_min'])
        self.Ks_max = float(setup['FLOW_CALIBRATION']['Ks_max'])
        self.psi0_min = float(setup['FLOW_CALIBRATION']['psi0_min'])
        self.psi0_max = float(setup['FLOW_CALIBRATION']['psi0_max'])
        self.pop = int(setup['FLOW_CALIBRATION']['pop'])
        self.gen = int(setup['FLOW_CALIBRATION']['gen'])


    def nash_sutcliffe_efficiency(self, observed, modeled):
        return 1 - np.sum((observed - modeled) ** 2) / np.sum((observed - np.mean(observed)) ** 2)

    def penalty(self, individual):
        if self.thetaR_min <= individual[0] <= self.thetaR_max:
            pen0 = 0
        else:
            pen0 = -10

        if self.thetaS_min <= individual[1] <= self.thetaS_max:
            pen1 = 0
        else:
            pen1 = -10

        if self.alpha_min <= individual[2] <= self.alpha_max:
            pen2 = 0
        else:
            pen2 = -10

        if self.n_min <= individual[3] <= self.n_max:
            pen3 = 0
        else:
            pen3 = -10

        if self.Ks_min <= individual[4] <= self.Ks_max:
            pen4 = 0
        else:
            pen4 = -10

        pen_total = pen0 + pen1 + pen2 + pen3 + pen4
        return pen_total

    def get_individual_values(self, individual):
        thetaR = individual[0]
        thetaS = individual[1]
        alpha = individual[2]
        n = individual[3]
        Ks = individual[4]
        return thetaR, thetaS, alpha, n, Ks

    def evaluate_by_nash_benchmark(self, individual):
        prec_input_file = "inputs/infiltration.dat"
        thetaR, thetaS, alpha, n, Ks = self.get_individual_values(individual)
        pars = {'thetaR': thetaR, 'thetaS': thetaS, 'alpha': alpha, 'n': n, 'Ks': Ks, 'neta': 0.5, 'Ss': 0.000001}
        calibration_input_file = "inputs/observed_benchmark.csv"

        wb = run_richards_benchmark.run_Richards(prec_input_file, pars)['S']
        wb = (wb - wb.min()) / 10

        penalty = self.penalty(individual)
        obs_df = pd.read_csv(calibration_input_file)['S']

        nash_outflow = self.nash_sutcliffe_efficiency(obs_df, wb)
        nash = nash_outflow + penalty
        return nash,

    def evaluate_by_nash_pp(self, individual, box_da=True):
        prec_pp = "inputs/rainfall_pp_filtered.csv"
        thetaR, thetaS, alpha, n, Ks, psi0 = self.get_individual_values(individual)
        pars = {'thetaR': thetaR, 'thetaS': thetaS, 'alpha': alpha, 'n': n, 'Ks': Ks, 'neta': 0.5, 'Ss': 0.000001}

        if box_da:
            calibration_input_file = "inputs/outflow_clipped_box_da.csv"
        else:
            calibration_input_file = "inputs/outflow_clipped_box_dc.csv"

        wb = run_richards_pp.run_Richards(prec_pp, pars)['S']
        wb = (wb - wb.min()) / 10
        penalty = self.penalty(individual)

        obs_df = pd.read_csv(calibration_input_file)['flow']
        obs_df = obs_df[:len(wb)]

        nash_outflow = self.nash_sutcliffe_efficiency(obs_df, wb)
        nash = nash_outflow + penalty
        return nash,
