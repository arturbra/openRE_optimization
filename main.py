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
    # individual = [0.131, 0.396, 0.423, 2.06, 3.52]
    # # pars = {'thetaR': 0.131, 'thetaS': 0.396, 'alpha': 0.423, 'n': 2.06, 'Ks': 3.32, 'psi0': 13.5, 'neta': 0.5, 'Ss': 0.000001}
    # pars = {'thetaR': individual[0], 'thetaS': individual[1], 'alpha': individual[2], 'n': individual[3], 'Ks': individual[4], 'neta': 0.5, 'Ss': 0.000001}
    # wb = run_richards_benchmark.run_Richards(prec_input_file, pars)
    # wb['S'] = (wb['S'] - min(wb['S'])) / 10
    #
    # calibration_input_file = "inputs/observed_benchmark.csv"
    # obs_df = pd.read_csv(calibration_input_file)['S']
    # # obs_df = obs_df[:len(wb)].shift(-2)
    # nash = run_BO.nash_sutcliffe_efficiency(obs_df, wb['S'])
    # print(nash)
    # plt.plot(wb['S'])
    # plt.plot(obs_df)
    # plt.show()

    #
    # Run for the PP

    # individual = [0.1298006671128676, 0.49009519861410955, 0.45312314660770364, 3.313737844977159, 5.570174014528901]
    # pars = {'thetaR': individual[0], 'thetaS': individual[1], 'alpha': individual[2], 'n': individual[3], 'Ks': individual[4], 'neta': 0.5, 'Ss': 0.000001}
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


    
    # ## GA calibration benchmark
    # setup_file = "inputs/setup_file.ini"
    # calibration = GA_model.Calibration(setup_file)
    #
    # toolbox = run_GA.initialize_toolbox(calibration, benchmark=True)
    # toolbox.register("map", futures.map)
    #
    # for s in range(1, 4):
    #     run_GA.water_flow_calibration(toolbox, calibration, s)

    # ## PSO benchmark
    # toolbox = run_PSO.create_toolbox()
    # toolbox.register("map", futures.map)
    # for s in range(1, 4):
    #     _, logbook, best_params = run_PSO.run_pso(s, toolbox)


    # ## BO benchmark
    # for s in range(2, 4):
    #     run_BO.run_bayesian_optimization(s, benchmark=True)

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


    # BO pp
    for s in range(1, 4):
        run_BO.run_bayesian_optimization(s, benchmark=False)

