import json
import matplotlib.pyplot as plt
import pandas as pd
import run_richards_benchmark
import run_richards_pp
import matplotlib.dates as mdates
import os
import re
import numpy as np
import seaborn as sns

def read_json_file_GA(file_name):
    generation_list = []
    individual_list = []
    fitness_list = []
    avg_fitness_list = []

    with open(file_name, 'r') as file:
        for line in file:
            data = json.loads(line)
            generation_list.append(data['generation'])
            individual_list.append(data['individual'])
            fitness_list.append(data['fitness'][0])
            avg_fitness_list.append(data['avg_fitness'])

    return generation_list, individual_list, fitness_list, avg_fitness_list



def plot_best_nash(iterations, best_nash, seed=0, save=False):
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, best_nash, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Best Nash')
    plt.title('Evolution of Best Nash Over Iterations')
    plt.grid()
    if save:
        plt.savefig(f'best_nash_seed{seed}.png')
    plt.show()


def plot_parameter_evolution(iterations, parameters, param_names, seed=0, save=False):
    num_params = len(parameters[0])
    num_rows = 2
    num_columns = 2
    plt.figure(figsize=(20, 5 * num_rows))

    for i in range(num_params):
        filtered_data = [(i, p) for i, p in zip(iterations, parameters) if p is not None]
        iterations, parameters = zip(*filtered_data)

        plt.subplot(num_rows, num_columns, i + 1)
        param_values = [p[i] for p in parameters]
        plt.plot(iterations, param_values, marker='o')
        plt.xlabel('Generation')
        plt.ylabel(param_names[i])
        plt.title(f'Evolution of {param_names[i]} Over Iterations')
        plt.grid()

    plt.tight_layout()

    if save:
        plt.savefig(f'parameter_evolution_seed{seed}.png')

    plt.show()


def create_combined_dataframe(individual):
    PREC_INPUT_FILE = "rainfall_pp_filtered.csv"
    rainfall = pd.read_csv(PREC_INPUT_FILE)['rain'] * 0.0254

    thetaR, thetaS, alpha, n, Ks, psi0 = individual
    pars = {'thetaR': thetaR, 'thetaS': thetaS, 'alpha': alpha, 'n': n, 'Ks': Ks, 'psi0': psi0, 'neta': 0.5, 'Ss': 0.000001}
    WB = run.run_Richards(PREC_INPUT_FILE, pars)['S']
    WB = (WB - WB.min()) / 10
    obs = pd.read_csv('outflow_clipped_box_da.csv')['flow']

    date = pd.read_csv('outflow_box_da.csv')['date']

    obs = obs[:len(WB)]
    date = date[:len(WB)]
    combined_dataframe = pd.DataFrame({'date': date, 'modeled': WB*1000, 'observed': obs*1000, 'rainfall': rainfall*1000})
    return combined_dataframe


def create_combined_dataframe_benchmark(individual):
    PREC_INPUT_FILE = r"C:\Users\ebz238\PycharmProjects\openRE\infiltrationproblem\input\infiltration.dat"
    rainfall = np.loadtxt(PREC_INPUT_FILE, skiprows=1, delimiter=',', usecols=1) / 1000
    thetaR, thetaS, alpha, n = individual
    pars = {'thetaR': thetaR, 'thetaS': thetaS, 'alpha': alpha, 'n': n, 'Ks': 0.0496, 'psi0': 3.59, 'neta': 0.5, 'Ss': 0.000001}
    WB = run_original.run_Richards(PREC_INPUT_FILE, pars)['QOUT']
    obs_file = r"C:\Users\ebz238\PycharmProjects\openRE\parameter_optimization\output.csv"
    obs = pd.read_csv(obs_file)['QOUT']
    combined_dataframe = pd.DataFrame({'modeled': WB*1000, 'observed': obs*1000, 'rainfall': rainfall*1000})
    return combined_dataframe


def plot_outflow_rainfall(combined_dataframe, nse, seed=0, save=False):
    plt.rcParams.update({
        'font.size': 14,         # general font size
        'axes.titlesize': 16,    # title font size
        'axes.labelsize': 14,    # axis label font size
        'xtick.labelsize': 12,   # x-axis tick label font size
        'ytick.labelsize': 12,   # y-axis tick label font size
        'legend.fontsize': 12,   # legend font size
    })
    # Convert 'date' column to datetime objects
    combined_dataframe['date'] = pd.to_datetime(combined_dataframe['date'])

    # Create a plot
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot modeled outflow and observed outflow on the first axis
    ax1.plot(combined_dataframe['date'], combined_dataframe['modeled'], label='Modeled Outflow', color='b', linestyle='-', linewidth=2)
    ax1.plot(combined_dataframe['date'], combined_dataframe['observed'], label='Observed Outflow', color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Outflow (mm)')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.76, 0.95))

    # Set the limits of the first axis based on the data range
    outflow_max = max(max(combined_dataframe['modeled']), max(combined_dataframe['observed']))
    ax1.set_ylim(0, 1.8 * outflow_max)

    # Create a second axis for precipitation
    ax2 = ax1.twinx()
    width = 1 / (24 * 60 / 5)  # Width corresponding to 5 minutes

    ax2.bar(combined_dataframe['date'], combined_dataframe['rainfall'], width=width, label='Observed Rainfall', color='black', alpha=0.6)
    ax2.set_ylabel('Precipitation (mm)')
    ax2.legend(loc='upper right')

    # Set the limits of the second axis based on the limits of the first axis
    rainfall_max = max(combined_dataframe['rainfall'])
    ax2.set_ylim(2.5 * rainfall_max, 0)

    # Configure the x-axis date formatting and tick spacing
    locator = mdates.AutoDateLocator(minticks=10, maxticks=15)
    formatter = mdates.DateFormatter('%H:%M')
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)

    # Rotate the x-axis labels for better legibility
    plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

    # Add a title to the plot

    plt.title('Permeable asphalt: Modeled Outflow, Observed Outflow, and Precipitation')
    ax1.annotate(f'NSE: {nse:.2f}', xy=(0.78, 0.82), xycoords='axes fraction', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    if save:
        plt.savefig(f'modeled_observed_seed{seed}.png')
    plt.show()


def plot_outflow_rainfall_benchmark(combined_dataframe, nse, seed=0, save=False):
    plt.rcParams.update({
        'font.size': 14,         # general font size
        'axes.titlesize': 16,    # title font size
        'axes.labelsize': 14,    # axis label font size
        'xtick.labelsize': 12,   # x-axis tick label font size
        'ytick.labelsize': 12,   # y-axis tick label font size
        'legend.fontsize': 12,   # legend font size
    })
    # Convert 'date' column to datetime objects
    time_indices = np.arange(len(combined_dataframe))


    # Create a plot
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot modeled outflow and observed outflow on the first axis
    ax1.plot(time_indices, combined_dataframe['modeled'], label='Modeled Outflow', color='b', linestyle='-', linewidth=2)
    ax1.plot(time_indices, combined_dataframe['observed'], label='Observed Outflow', color='r', linestyle='--', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Outflow (mm)')
    ax1.legend(loc='upper left', bbox_to_anchor=(0.76, 0.95))

    # Set the limits of the first axis based on the data range
    outflow_max = max(max(combined_dataframe['modeled']), max(combined_dataframe['observed']))
    ax1.set_ylim(0, 1.8 * outflow_max)

    # Create a second axis for precipitation
    ax2 = ax1.twinx()

    ax2.bar(time_indices, combined_dataframe['rainfall'], label='Observed Rainfall', color='black', alpha=0.6)
    ax2.set_ylabel('Precipitation (mm)')
    ax2.legend(loc='upper right')

    # Set the limits of the second axis based on the limits of the first axis
    rainfall_max = max(combined_dataframe['rainfall'])
    ax2.set_ylim(2.5 * rainfall_max, 0)

    # Add a title to the plot
    plt.title('Benchmark GA: Modeled Outflow, Observed Outflow, and Precipitation')
    ax1.annotate(f'NSE: {nse:.2f}', xy=(0.78, 0.82), xycoords='axes fraction', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    if save:
        plt.savefig(f'modeled_observed_seed{seed}.png')
    plt.show()


def plot_fitness_vs_generation(generation_list, fitness_list1, fitness_list2, fitness_list3):
    sns.set(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 6), sharex=True)

    # Function to add the best fitness annotation
    def add_best_fitness_annotation(ax, fitness_list, xpos, ypos):
        best_fitness = fitness_list[-1]
        ax.annotate(f"Best Nash: {best_fitness:.3f}",
                    xy=(xpos, ypos),
                    xycoords='axes fraction',
                    fontsize=12,
                    color='black',
                    bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=0.5'))

    # Plot fitness on the first subplot
    ax1.plot(generation_list, fitness_list1, label="Fitness seed 1", marker='o', markersize=5, linestyle='-', linewidth=2, color='tab:blue')
    ax1.set_xlabel("Generation", fontsize=14, labelpad=15)
    ax1.set_ylabel("Nash", fontsize=14, labelpad=15)

    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', labelsize=12)
    add_best_fitness_annotation(ax1, fitness_list1, 0.65, 0.15)

    # Plot fitness on the second subplot
    ax2.plot(generation_list, fitness_list2, label="Fitness seed 2", marker='o', markersize=5, linestyle='-', linewidth=2, color='tab:orange')
    ax2.set_xlabel("Generation", fontsize=14, labelpad=15)
    ax2.set_ylabel("Nash", fontsize=14, labelpad=15)
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    add_best_fitness_annotation(ax2, fitness_list2, 0.65, 0.15)

    # Plot fitness on the third subplot
    ax3.plot(generation_list, fitness_list3, label="Fitness seed 3", marker='o', markersize=5, linestyle='-', linewidth=2, color='tab:green')
    ax3.set_xlabel("Generation", fontsize=14, labelpad=15)
    ax3.set_ylabel("Nash", fontsize=14, labelpad=15)
    ax3.legend(fontsize=12)
    ax3.tick_params(axis='both', labelsize=12)
    add_best_fitness_annotation(ax3, fitness_list3, 0.65, 0.15)

    sns.despine(left=True, bottom=True)
    plt.show()


def plot_avg_fitness_vs_generation(generation_list, fitness_list1, fitness_list2, fitness_list3):
    sns.set(style="whitegrid")
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 6), sharex=True)

    # Plot fitness on the first subplot
    ax1.plot(generation_list, fitness_list1, label="Avg. Fitness seed 1", marker='o', markersize=5, linestyle='-', linewidth=2, color='tab:blue')
    ax1.set_xlabel("Generation", fontsize=14, labelpad=15)
    ax1.set_ylabel("Nash", fontsize=14, labelpad=15)

    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', labelsize=12)

    # Plot fitness on the second subplot
    ax2.plot(generation_list, fitness_list2, label="Avg. Fitness seed 2", marker='o', markersize=5, linestyle='-', linewidth=2, color='tab:orange')
    ax2.set_xlabel("Generation", fontsize=14, labelpad=15)
    ax2.set_ylabel("Nash", fontsize=14, labelpad=15)
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)

    # Plot fitness on the third subplot
    ax3.plot(generation_list, fitness_list3, label="Avg. Fitness seed 3", marker='o', markersize=5, linestyle='-', linewidth=2, color='tab:green')
    ax3.set_xlabel("Generation", fontsize=14, labelpad=15)
    ax3.set_ylabel("Nash", fontsize=14, labelpad=15)
    ax3.legend(fontsize=12)
    ax3.tick_params(axis='both', labelsize=12)

    sns.despine(left=True, bottom=True)
    plt.show()

# BO Asphalt

# seed = 1
# file_path = fr"C:\Users\ebz238\PycharmProjects\richards_pp_observed\Results\Asphalt\BO\JSON\bayesian_optimization_seed{seed}_box_da.json"
# iterations, parameters, best_parameters, best_nash = read_results(file_path)
#
# max_value = max(best_nash)
# max_index = best_nash.index(max_value)
# best = best_parameters[max_index]
# print(f"Theta R: {best[0]: .3f} \n"
#       f"Theta S: {best[1]: .3f} \n"
#       f"Alpha: {best[2]: .3f}\n"
#       f"n: {best[3]: .3f}\n"
#       f"Ks: {best[4]} \n"
#       f"Psi0: {best[5]}")
#
# plot_best_nash(iterations, best_nash, seed=seed, save=True)
# param_names = [r'$\theta_R$', r'$\theta_S$', r'$\alpha$', r'$n$', r'$K_s$', r'$\psi_0$']  # Replace with actual parameter names
# plot_parameter_evolution(iterations, parameters, param_names, seed=seed, save=True)
#
# individual = best_parameters[-2]
#
# combined_dataframe = create_combined_dataframe(individual)
#
# plot_outflow_rainfall(combined_dataframe, seed=seed, nse=best_nash[-2], save=True)


# GA Asphalt
file_path_1 = f"outputs/GA/Box_DA/GA_seed1_DA.json"

generation_1, individual_1, fitness_1, avg_fitness_1 = read_json_file_GA(file_path_1)

file_path_2 = f"outputs/GA/Box_DA/GA_seed2_DA.json"

generation_2, individual_2, fitness_2, avg_fitness_2 = read_json_file_GA(file_path_2)

file_path_3 = f"outputs/GA/Box_DA/GA_seed3_DA.json"

generation_3, individual_3, fitness_3, avg_fitness_3 = read_json_file_GA(file_path_3)

plot_fitness_vs_generation(generation_1, fitness_1, fitness_2, fitness_3)
plot_avg_fitness_vs_generation(generation_1, avg_fitness_1, avg_fitness_2, avg_fitness_3)





# # GA Benchmark extraction
# seed = 1
# directory = r"C:\Users\ebz238\PycharmProjects\richards_pp_observed\Results\Benchmark\GA\JSON"
# info = process_json_files(directory)
# individual_seed_1 = info[seed]['individuals'][:101]
# generations_seed_1 = info[seed]['generations'][:101]
# nash_seed_1 = info[seed]['fitness'][:101]
#
# max_value = max(nash_seed_1)
# max_index = nash_seed_1.index(max_value)
# best = individual_seed_1[max_index]
# print(f"Theta R: {best[0]: .3f} \n"
#       f"Theta S: {best[1]: .3f} \n"
#       f"Alpha: {best[2]: .3f}\n"
#       f"n: {best[3]: .3f}")
#
# param_names = [r'$\theta_R$', r'$\theta_S$', r'$\alpha$', r'$n$']
# plot_parameter_evolution(generations_seed_1, individual_seed_1, param_names, seed=seed, save=True)
# plot_best_nash(generations_seed_1, nash_seed_1, seed=seed, save=True)
# combined_dataframe = create_combined_dataframe_benchmark(individual_seed_1[-1])
# plot_outflow_rainfall_benchmark(combined_dataframe, nash_seed_1[-1], seed=seed, save=True)


# PSO Benchmark extraction
# seed = 1
# file_path = rf"C:\Users\ebz238\PycharmProjects\richards_pp_observed\Results\Benchmark\PSO\JSON\logbook_seed_{seed}.json"
# generations, best_particles, max_values = extract_data_from_json(file_path)
#
# max_value = max(max_values)
# max_index = max_values.index(max_value)
# best = best_particles[max_index]
# print(f"Theta R: {best[0]: .3f} \n"
#       f"Theta S: {best[1]: .3f} \n"
#       f"Alpha: {best[2]: .3f}\n"
#       f"n: {best[3]: .3f}")
#
# param_names = [r'$\theta_R$', r'$\theta_S$', r'$\alpha$', r'$n$']
# plot_parameter_evolution(generations, best_particles, param_names, seed=seed, save=True)
# plot_best_nash(generations, max_values, seed=seed, save=True)
# combined_dataframe = create_combined_dataframe_benchmark(best_particles[-1])
# plot_outflow_rainfall_benchmark(combined_dataframe, max(max_values), seed=seed, save=True)



# BO Benchmark extraction
# seed = 1
# file_path = rf"C:\Users\ebz238\PycharmProjects\richards_pp_observed\Results\Benchmark\BO\JSON\bayesian_optimization_seed{seed}.json"
#
# iterations, best_parameters, best_nash = extract_data_from_json_BO(file_path)
#
# max_value = max(best_nash)
# max_index = best_nash.index(max_value)
# best = best_parameters[max_index]
# print(f"Theta R: {best[0]: .3f} \n"
#       f"Theta S: {best[1]: .3f} \n"
#       f"Alpha: {best[2]: .3f}\n"
#       f"n: {best[3]: .3f}")
#
# param_names = [r'$\theta_R$', r'$\theta_S$', r'$\alpha$', r'$n$']
# plot_parameter_evolution(iterations, best_parameters, param_names, seed=seed, save=True)
# plot_best_nash(iterations, best_nash, seed=seed, save=True)
# combined_dataframe = create_combined_dataframe_benchmark(best_parameters[-1])
# plot_outflow_rainfall_benchmark(combined_dataframe, best_nash[-1], seed=seed, save=True)



