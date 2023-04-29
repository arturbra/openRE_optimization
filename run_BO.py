import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
import run_richards_benchmark
import run_richards_pp
import time
import json


def custom_callback(res, seed):
    current_params = res.x_iters[-1]
    # current_modeled = run_Richards(PREC_INPUT_FILE, get_param_dict(current_params))['S']
    # current_modeled = (current_modeled - current_modeled.min()) / 10
    # current_nse = nash_sutcliffe_efficiency(observed_outflow, current_modeled)
    iteration = len(res.func_vals)

    # Find the best NSE so far and its corresponding parameters
    best_nse_idx = np.argmin(res.func_vals)
    best_params_so_far = res.x_iters[best_nse_idx]
    best_nse_so_far = -res.func_vals[best_nse_idx]

    result_dict = {
        "iteration": iteration,
        "parameters": current_params,
        # "nse": current_nse,
        "best_parameters": best_params_so_far,
        "best_nse": best_nse_so_far
    }

    json_filename = f"bayesian_optimization_seed{seed}_box_dc.json"
    result_json_string = json.dumps(result_dict)
    with open(json_filename, "a") as outfile:
        outfile.write(result_json_string + '\n')

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
    PREC_INPUT_FILE = "rainfall_pp_filtered.csv"
    observed_outflow = pd.read_csv('inputs/outflow_clipped_box_dc.csv')['flow']
    param = {'thetaR': pars[0], 'thetaS': pars[1], 'alpha': pars[2], 'n': pars[3], 'Ks': pars[4], 'psi0': pars[5],
            'neta': 0.5, 'Ss': 0.000001}
    modeled_outflow = run_richards_benchmark.run_Richards(PREC_INPUT_FILE, param)['S']
    modeled_outflow = (modeled_outflow - modeled_outflow.min()) / 10
    nse = nash_sutcliffe_efficiency(observed_outflow, modeled_outflow)
    return -nse  # Minimize the negative of NSE to maximize NSE


def objective_function_pp(pars):
    PREC_INPUT_FILE = "inputs/rainfall_pp_filtered.csv"
    obs_file = "inputs/outflow_clipped_box_dc.csv"
    observed_outflow = pd.read_csv(obs_file)['flow']
    param = {'thetaR': pars[0], 'thetaS': pars[1], 'alpha': pars[2], 'n': pars[3], 'Ks': pars[4], 'psi0': pars[5],
            'neta': 0.5, 'Ss': 0.000001}
    modeled_outflow = run_richards_pp.run_Richards(PREC_INPUT_FILE, param)['S']
    modeled_outflow = (modeled_outflow - modeled_outflow.min()) / 10
    nse = nash_sutcliffe_efficiency(observed_outflow, modeled_outflow)
    return -nse  # Minimize the negative of NSE to maximize NSE


def run_bayesian_optimization(seed):
    search_space = [
        Real(0.1, 0.2, name="theta_r"),
        Real(0.3, 0.6, name="theta_s"),
        Real(0, 1, name="alpha"),
        Real(1.1, 10, name="n"),
        Real(0.001, 50, name="Ks"),
        Real(-10, 10, name='psi0')
    ]

    # Run Bayesian Optimization
    start_time = time.time()

    result = gp_minimize(func=objective_function_pp,
                         dimensions=search_space,
                         n_calls=500,
                         random_state=seed,
                         callback=[lambda res: custom_callback(res, seed)])

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    t_time = {"total time": total_time}
    time_json_string = json.dumps(t_time)
    json_filename = f"bayesian_optimization_seed{seed}_box_dc.json"
    # Get the best parameters
    best_params = result.x
    best_nse = -result.fun

    # Save results to a JSON file
    with open(json_filename, "a") as outfile:
        outfile.write(time_json_string + '\n')

    print("Best parameters:", best_params)
    print("Best NSE:", best_nse)

