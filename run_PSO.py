import numpy as np
import random
import json
from deap import base, creator, tools, algorithms
import sys
import pandas as pd
import configparser
import json
import time

import run_richards_benchmark
import run_richards_pp


def nash_sutcliffe_efficiency(observed, modeled):
    return 1 - np.sum((observed - modeled) ** 2) / np.sum((observed - np.mean(observed)) ** 2)


def evaluate_pp(particle, box_da=False):
    pars = {'thetaR': particle[0], 'thetaS': particle[1], 'alpha': particle[2], 'n': particle[3], 'Ks': particle[4], 'neta': 0.5, 'Ss': 0.000001}
    prec_input_file = "inputs/rainfall_pp_filtered.csv"

    if box_da:
        calibration_input_file = "inputs/outflow_clipped_box_da.csv"

    else:
        calibration_input_file = "inputs/outflow_clipped_box_dc.csv"

    modeled = run_richards_pp.run_Richards(prec_input_file, pars)['S']
    modeled = (modeled - modeled.min()) / 10

    observed_outflow = pd.read_csv(calibration_input_file)['flow']

    return nash_sutcliffe_efficiency(observed_outflow, modeled),


def evaluate_benchmark(particle):
    pars = {'thetaR': particle[0], 'thetaS': particle[1], 'alpha': particle[2], 'n': particle[3], 'Ks': particle[4], 'neta': 0.5, 'Ss': 0.000001}
    prec_input_file = "inputs/infiltration.dat"
    calibration_input_file = "inputs/observed_benchmark.csv"
    observed_outflow = pd.read_csv(calibration_input_file)['S']
    modeled = run_richards_benchmark.run_Richards(prec_input_file, pars)['S']
    modeled = (modeled - modeled.min()) / 10

    return nash_sutcliffe_efficiency(observed_outflow, modeled),


def evaluate(particle, benchmark=True, box_da=False):
    pars = {'thetaR': particle[0], 'thetaS': particle[1], 'alpha': particle[2], 'n': particle[3], 'Ks': particle[4], 'neta': 0.5, 'Ss': 0.000001}
    if benchmark:
        prec_input_file = "inputs/infiltration.dat"
        calibration_input_file = "inputs/observed_benchmark.csv"
        observed_outflow = pd.read_csv(calibration_input_file)['S']
        modeled = run_richards_benchmark.run_Richards(prec_input_file, pars)['S']
        modeled = (modeled - modeled.min()) / 10

    else:
        prec_input_file = "inputs/rainfall_pp_filtered.csv"

        if box_da:
            calibration_input_file = "inputs/outflow_clipped_box_da.csv"

        else:
            calibration_input_file = "inputs/outflow_clipped_box_dc.csv"

        modeled = run_richards_pp.run_Richards(prec_input_file, pars)['S']
        modeled = (modeled - modeled.min()) / 10

        observed_outflow = pd.read_csv(calibration_input_file)['flow']

    return nash_sutcliffe_efficiency(observed_outflow, modeled),


def create_particle():
    setup = configparser.ConfigParser()
    setup_file = "inputs/setup_file.ini"
    setup.read(setup_file)
    thetaR_min = float(setup['FLOW_CALIBRATION']['thetaR_min'])
    thetaR_max = float(setup['FLOW_CALIBRATION']['thetaR_max'])
    thetaS_min = float(setup['FLOW_CALIBRATION']['thetaS_min'])
    thetaS_max = float(setup['FLOW_CALIBRATION']['thetaS_max'])
    alpha_min = float(setup['FLOW_CALIBRATION']['alpha_min'])
    alpha_max = float(setup['FLOW_CALIBRATION']['alpha_max'])
    n_min = float(setup['FLOW_CALIBRATION']['n_min'])
    n_max = float(setup['FLOW_CALIBRATION']['n_max'])
    Ks_min = float(setup['FLOW_CALIBRATION']['ks_min'])
    Ks_max = float(setup['FLOW_CALIBRATION']['ks_max'])

    pmin = [thetaR_min, thetaS_min, alpha_min, n_min, Ks_min]
    pmax = [thetaR_max, thetaS_max, alpha_max, n_max, Ks_max]
    scaling_factor = 0.2
    smin = [-scaling_factor * (pmax[i] - pmin[i]) for i in range(len(pmin))]
    smax = [scaling_factor * (pmax[i] - pmin[i]) for i in range(len(pmax))]

    part = creator.Particle([random.uniform(pmin[i], pmax[i]) for i in range(5)])
    part.speed = [random.uniform(smin[i], smax[i]) for i in range(5)]
    part.smin = smin
    part.smax = smax
    return part


def update_particle(part, best, phi1, phi2, w=0.7298):
    setup = configparser.ConfigParser()
    setup_file = "inputs/setup_file.ini"
    setup.read(setup_file)
    thetaR_min = float(setup['FLOW_CALIBRATION']['thetaR_min'])
    thetaR_max = float(setup['FLOW_CALIBRATION']['thetaR_max'])
    thetaS_min = float(setup['FLOW_CALIBRATION']['thetaS_min'])
    thetaS_max = float(setup['FLOW_CALIBRATION']['thetaS_max'])
    alpha_min = float(setup['FLOW_CALIBRATION']['alpha_min'])
    alpha_max = float(setup['FLOW_CALIBRATION']['alpha_max'])
    Ks_min = float(setup['FLOW_CALIBRATION']['ks_min'])
    Ks_max = float(setup['FLOW_CALIBRATION']['ks_max'])
    n_min = float(setup['FLOW_CALIBRATION']['n_max'])
    n_max = float(setup['FLOW_CALIBRATION']['n_max'])

    pmin = [thetaR_min, thetaS_min, alpha_min, n_min, Ks_min]
    pmax = [thetaR_max, thetaS_max, alpha_max, n_max, Ks_max]

    u1 = (random.uniform(0, phi1) for _ in range(len(part)))
    u2 = (random.uniform(0, phi2) for _ in range(len(part)))
    v_u1 = map(lambda x: x[0] * (part.best[x[1]] - part[x[1]]), zip(u1, range(len(part))))
    v_u2 = map(lambda x: x[0] * (best[x[1]] - part[x[1]]), zip(u2, range(len(part))))

    part.speed = list(map(lambda x: w * x[0] + x[1] + x[2], zip(part.speed, v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin[i]:
            part.speed[i] = np.sign(speed) * part.smin[i]
        elif abs(speed) > part.smax[i]:
            part.speed[i] = np.sign(speed) * part.smax[i]

    # Update the position and enforce bounds
    for i in range(len(part)):
        new_pos = part[i] + part.speed[i]

        if new_pos < pmin[i]:
            if i == 3:
                part[i] = random.uniform(n_min, n_max)
            else:
                part[i] = pmin[i]
        elif new_pos > pmax[i]:
            part[i] = pmax[i]
        else:
            part[i] = new_pos


def create_toolbox():
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list,
                   smin=None, smax=None, best=None)
    toolbox = base.Toolbox()
    toolbox.register("particle", create_particle)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", update_particle, phi1=2.0, phi2=2.0)
    toolbox.register("evaluate", evaluate)

    return toolbox


# def save_logbook_to_json(logbook, seed):
#     logbook_data = []
#     filename = f"logbook_seed_{seed}.json"
#     for record in logbook:
#         logbook_data.append({
#             "gen": record["gen"],
#             "evals": record["evals"],
#             "time": record['time'],
#             'best_particle': record['best_particle'],
#             "avg": record["avg"],
#             "min": record["min"],
#             "max": record["max"]
#         })
#
#     with open(filename, "w") as outfile:
#         json.dump(logbook_data, outfile)


def run_pso(toolbox, s):
    random.seed(s)
    setup = configparser.ConfigParser()
    setup_file = "inputs/setup_file.ini"
    setup.read(setup_file)
    pop = int(setup['FLOW_CALIBRATION']['pop'])
    gen = int(setup['FLOW_CALIBRATION']['gen'])
    pop = toolbox.population(n=pop)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals", "time", "best_particle", "best_fitness"] + stats.fields

    best = None
    print("Calibration started:")

    logbook_data = []
    for g in range(gen):
        tic = time.time()
        print("=========================")
        print(f"Generation {g}")
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        print(f"Best particle: {best} \nBest fitness: {best.fitness.values}")
        runtime = time.time() - tic
        gen_time = runtime / 60
        time_left = runtime * (gen - g) / 60
        print('approx time left = %.2f minutes' % (time_left))
        logbook.record(gen=g, evals=len(pop), time=gen_time, best_particle=best, **stats.compile(pop))

        print(logbook.stream)
        logbook_data.append({
            "gen": g,
            "evals": len(pop),
            "time": gen_time,
            'best_particle': best,
            'best_fitness': best.fitness.values,
            "avg": stats.compile(pop)["avg"],
            "min": stats.compile(pop)["min"],
            "max": stats.compile(pop)["max"]
        })

        filename = f"PSO_seed_{s}.json"
        with open(filename, "w") as outfile:
            json.dump(logbook_data, outfile)

    return pop, logbook, best
