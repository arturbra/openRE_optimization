import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
import run_richards_benchmark
import run_richards_pp
import time
import json
import configparser

def custom_callback(res, all_results):
    current_params = res.x_iters[-1]
    iteration = len(res.func_vals)

    # Find the best NSE so far and its corresponding parameters
    best_nse_idx = np.argmin(res.func_vals)
    best_params_so_far = res.x_iters[best_nse_idx]
    best_nse_so_far = -res.func_vals[best_nse_idx]

    result_dict = {
        "iteration": iteration,
        "parameters": current_params,
        "best_parameters": best_params_so_far,
        "best_nse": best_nse_so_far
    }

    all_results.append(result_dict)
    print(f"Iteration {iteration}")

    if iteration % 10 == 0:
        print(f"Best parameters so far: {best_params_so_far}, Best NSE so far: {best_nse_so_far}")
        print(f"Iteration: {iteration}")


def get_param_dict(pars):
    return {'thetaR': pars[0], 'thetaS': pars[1], 'alpha': pars[2], 'n': pars[3], 'Ks': pars[4], 'psi0': pars[5],
            'neta': 0.5, 'Ss': 0.000001}


def nash_sutcliffe_efficiency(observed, modeled):
    return 1 - np.sum((observed - modeled) ** 2) / np.sum((observed - np.mean(observed)) ** 2)


def objective_function_benchmark(pars):
    prec_input_file = "inputs/infiltration.dat"
    observed_outflow = pd.read_csv('inputs/observed_benchmark.csv')['S']
    param = {'thetaR': pars[0], 'thetaS': pars[1], 'alpha': pars[2], 'n': pars[3], 'Ks': pars[4],
            'neta': 0.5, 'Ss': 0.000001}
    modeled_outflow = run_richards_benchmark.run_Richards(prec_input_file, param)['S']
    modeled_outflow = (modeled_outflow - modeled_outflow.min()) / 10
    nse = nash_sutcliffe_efficiency(observed_outflow, modeled_outflow)
    return -nse  # Minimize the negative of NSE to maximize NSE


def objective_function_pp(pars):
    PREC_INPUT_FILE = "inputs/rainfall_pp_filtered.csv"
    obs_file = "inputs/outflow_clipped_box_dc.csv"
    observed_outflow = pd.read_csv(obs_file)['flow']
    param = {'thetaR': pars[0], 'thetaS': pars[1], 'alpha': pars[2], 'n': pars[3], 'Ks': pars[4],
            'neta': 0.5, 'Ss': 0.000001}
    modeled_outflow = run_richards_pp.run_Richards(PREC_INPUT_FILE, param)['S']
    modeled_outflow = (modeled_outflow - modeled_outflow.min()) / 10
    nse = nash_sutcliffe_efficiency(observed_outflow, modeled_outflow)
    return -nse  # Minimize the negative of NSE to maximize NSE


def run_bayesian_optimization(seed, benchmark=True):
    setup_file = 'inputs/setup_file.ini'

    setup = configparser.ConfigParser()
    setup.read(setup_file)
    thetaR_min = float(setup['FLOW_CALIBRATION']['thetaR_min'])
    thetaR_max = float(setup['FLOW_CALIBRATION']['thetaR_max'])
    thetaS_min = float(setup['FLOW_CALIBRATION']['thetaS_min'])
    thetaS_max = float(setup['FLOW_CALIBRATION']['thetaS_max'])
    alpha_min = float(setup['FLOW_CALIBRATION']['alpha_min'])
    alpha_max = float(setup['FLOW_CALIBRATION']['alpha_max'])
    n_min = float(setup['FLOW_CALIBRATION']['n_min'])
    n_max = float(setup['FLOW_CALIBRATION']['n_max'])
    Ks_min = float(setup['FLOW_CALIBRATION']['Ks_min'])
    Ks_max = float(setup['FLOW_CALIBRATION']['Ks_max'])

    search_space = [
        Real(thetaR_min, thetaR_max, name="theta_r"),
        Real(thetaS_min, thetaS_max, name="theta_s"),
        Real(alpha_min, alpha_max, name="alpha"),
        Real(n_min, n_max, name="n"),
        Real(Ks_min, Ks_max, name="Ks")
    ]

    # Run Bayesian Optimization
    all_results = []
    start_time = time.time()
    if benchmark:
        result = gp_minimize(func=objective_function_benchmark,
                             dimensions=search_space,
                             n_calls=500,
                             random_state=seed,
                             callback=[lambda res: custom_callback(res, all_results)])
    else:
        result = gp_minimize(func=objective_function_pp,
                             dimensions=search_space,
                             n_calls=500,
                             random_state=seed,
                             callback=[lambda res: custom_callback(res, all_results)])

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print(f'Total time: {total_time: .2f} minutes')
    json_filename = f"bayesian_optimization_seed{seed}_benchmark.json"
    # Get the best parameters
    best_params = result.x
    best_nse = -result.fun

    # Save results to a JSON file
    with open(json_filename, "w") as outfile:
        json.dump(all_results, outfile)

    print("Best parameters:", best_params)
    print("Best NSE:", best_nse)


