import run_richards_benchmark
import run_richards_pp
import run_GA
import GA_model
from scoop import futures
import matplotlib.pyplot as plt
import pandas as pd
import run_PSO
import run_BO


if __name__ == '__main__':
    # #Run for the benchmark
    # prec_input_file = "inputs/infiltration.dat"
    # pars = {'thetaR': 0.1, 'thetaS': 0.2, 'alpha': 0.5, 'n': 2, 'Ks': 0.04, 'psi0': 3.5, 'neta': 0.5, 'Ss': 0.000001}
    # wb = run_richards_benchmark.run_Richards(prec_input_file, pars)
    #
    #
    #Run for the PP

    #
    # individual = [0.11794939522713252, 0.5014643011668757, 0.5697775404333506, 10.0, 0.9071343643361653, 9.964788249998058]
    # pars = {'thetaR': individual[0], 'thetaS': individual[1], 'alpha': individual[2], 'n': individual[3], 'Ks': individual[4], 'psi0': individual[5], 'neta': 0.5, 'Ss': 0.000001}
    #
    # prec_pp = "inputs/rainfall_pp_filtered.csv"
    # wb = run_richards_pp.run_Richards(prec_pp, pars)['S']
    # wb = (wb - wb.min()) / 10
    #
    # calibration_input_file = "inputs/outflow_clipped_box_dc.csv"
    # obs_df = pd.read_csv(calibration_input_file)['flow']
    # obs_df = obs_df[:len(wb)].shift(-2)
    # plt.plot(wb)
    # plt.plot(obs_df)
    # plt.show()
    #

    
    # ## GA calibration benchmark
    # setup_file = "inputs/setup_file.ini"
    # calibration = GA_model.Calibration(setup_file)
    #
    # toolbox = run_GA.initialize_toolbox(calibration, benchmark=True)
    # toolbox.register("map", futures.map)
    #
    # for s in range(1, 4):
    #     run_GA.water_flow_calibration(toolbox, calibration, s)

    ## PSO benchmark
    toolbox = run_PSO.create_toolbox()
    toolbox.register("map", futures.map)
    for s in range(1, 4):
        _, logbook, best_params = run_PSO.run_pso(toolbox, s)

    # ## GA calibration pp
    # setup_file = "inputs/setup_file.ini"
    # calibration = GA_model.Calibration(setup_file)
    #
    # toolbox = run_GA.initialize_toolbox(calibration, benchmark=False)
    # toolbox.register("map", futures.map)
    #
    # for s in range(1, 4):
    #     run_GA.water_flow_calibration(toolbox, calibration, s)

    # # PSO pp
    # toolbox = run_PSO.create_toolbox()
    # toolbox.register("map", futures.map)
    # for s in range(1, 4):
    #     _, logbook, best_params = run_PSO.run_pso(toolbox, s)


    # # BO pp
    # for s in range(1, 4):
    #     run_BO.run_bayesian_optimization(s)

