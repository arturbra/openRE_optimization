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

def read_json_file_GA_pp(file_name):
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


def read_json_file_GA_benchmark(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)

    individuals = []
    generations = []
    fitness_values = []
    avg_fitness_values = []

    for entry in data:
        individuals.append(entry['individual'])
        generations.append(entry['generation'])
        fitness_values.append(entry['fitness'][0])  # Assuming fitness is always a list with a single value
        avg_fitness_values.append(entry['avg_fitness'])

    return generations, individuals, fitness_values, avg_fitness_values


def read_json_file_PSO_benchmark(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)

    generations = []
    best_particles = []
    best_fitnesses = []
    averages = []

    for entry in data:
        generations.append(entry['gen'])
        best_particles.append(entry['best_particle'])
        best_fitnesses.append(entry['best_fitness'][0])
        averages.append(entry['avg'])

    return generations, best_particles, best_fitnesses, averages


def read_json_file_PSO_pp(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)

    generations = []
    best_particles = []
    best_fitnesses = []
    averages = []

    for entry in data:
        generations.append(entry['gen'])
        best_particles.append(entry['best_particle'])
        best_fitnesses.append(entry['max'])
        averages.append(entry['avg'])

    return generations, best_particles, best_fitnesses, averages


def read_json_file_BO_benchmark_1(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()

    iterations = []
    best_parameters = []
    best_nses = []

    for line in lines:
        entry = json.loads(line)
        iterations.append(entry['iteration'])
        best_parameters.append(entry['best_parameters'])
        best_nses.append(entry['best_nse'])

    return iterations, best_parameters, best_nses


def read_json_file_BO_benchmark(file_name):
    with open(file_name, 'r') as file:
        data = json.load(file)

    iteration = []
    best_parameters = []
    best_nses = []
    avg_fitness_values = []

    for entry in data:
        iteration.append(entry['iteration'])
        best_parameters.append(entry['best_parameters'])
        best_nses.append(entry['best_nse'])

    return iteration, best_parameters, best_nses

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


def create_combined_dataframe(individual, box_da=True):
    PREC_INPUT_FILE = "inputs/rainfall_pp_filtered.csv"
    rainfall = pd.read_csv(PREC_INPUT_FILE)['rain'] * 0.0254

    thetaR, thetaS, alpha, n, Ks, psi0 = individual
    pars = {'thetaR': thetaR, 'thetaS': thetaS, 'alpha': alpha, 'n': n, 'Ks': Ks, 'psi0': psi0, 'neta': 0.5, 'Ss': 0.000001}
    WB = run_richards_pp.run_Richards(PREC_INPUT_FILE, pars)['S']
    WB = (WB - WB.min()) / 10

    if box_da:
        obs = pd.read_csv('inputs/outflow_clipped_box_da.csv')['flow']
    else:
        obs = pd.read_csv('inputs/outflow_clipped_box_dc.csv')['flow']
        obs = obs.shift(-3)

    date = pd.read_csv('inputs/outflow_box_da.csv')['date']

    obs = obs[:len(WB)]
    date = date[:len(WB)]
    combined_dataframe = pd.DataFrame({'date': date, 'modeled': WB*1000, 'observed': obs*1000, 'rainfall': rainfall*1000})
    return combined_dataframe


def create_combined_dataframe_benchmark(individual):
    PREC_INPUT_FILE = "inputs/infiltration.dat"
    rainfall = np.loadtxt(PREC_INPUT_FILE, skiprows=1, delimiter=',', usecols=1) / 1000
    rainfall = rainfall[:int(len(rainfall) / 4)]
    thetaR, thetaS, alpha, n, Ks = individual
    pars = {'thetaR': thetaR, 'thetaS': thetaS, 'alpha': alpha, 'n': n, 'Ks': Ks, 'neta': 0.5, 'Ss': 0.000001}
    WB = run_richards_benchmark.run_Richards(PREC_INPUT_FILE, pars)['S']
    WB = (WB - WB.min()) / 10
    obs_file = r"inputs/observed_benchmark.csv"
    obs = pd.read_csv(obs_file)['S']

    modeled = np.array(WB) * 1000
    observed = np.array(obs) * 1000
    rainfall = np.array(rainfall) * 1000

    max_length = max(len(modeled), len(observed), len(rainfall))

    modeled_padded = np.pad(modeled, (0, max_length - len(modeled)), constant_values=np.nan)
    observed_padded = np.pad(observed, (0, max_length - len(observed)), constant_values=np.nan)
    rainfall_padded = np.pad(rainfall, (0, max_length - len(rainfall)), constant_values=np.nan)

    combined_dataframe = pd.DataFrame({'modeled': modeled_padded, 'observed': observed_padded, 'rainfall': rainfall_padded})
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


def plot_outflow_rainfall_benchmark(combined_dataframe, nse, seed=0, save=False, filename=''):
    combined_dataframe = combined_dataframe.iloc[20:, :]
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

    ax2.bar(time_indices, combined_dataframe['rainfall'], label='Observed Rainfall', color='black', alpha=1)
    ax2.set_ylabel('Precipitation (mm)')
    ax2.legend(loc='upper right')

    # Set the limits of the second axis based on the limits of the first axis
    rainfall_max = max(combined_dataframe['rainfall'])
    ax2.set_ylim(2.5 * rainfall_max, 0)

    # Add a title to the plot
    plt.title('Benchmark GA: Modeled Outflow, Observed Outflow, and Precipitation')
    ax1.annotate(f'NSE: {nse:.2f}', xy=(0.78, 0.82), xycoords='axes fraction', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    if save:
        plt.savefig(f'{filename}_seed_{seed}.png')
    plt.show()


def plot_fitness_vs_generation(generation_list, fitness_list1, fitness_list2, fitness_list3, save=False, filename=''):
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
    if save:
        plt.savefig(filename)
    plt.show()


def plot_avg_fitness_vs_generation(generation_list, fitness_list1, fitness_list2, fitness_list3, save=True, filename=""):
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
    if save:
        plt.savefig(filename)
    plt.show()
#
# #
# # GA Box_DA
# file_path_1 = f"outputs/GA/Box_DA/GA_seed1_DA.json"
#
# generation_1, individual_1, fitness_1, avg_fitness_1 = read_json_file_GA_pp(file_path_1)
#
# file_path_2 = f"outputs/GA/Box_DA/GA_seed2_DA.json"
#
# generation_2, individual_2, fitness_2, avg_fitness_2 = read_json_file_GA_pp(file_path_2)
#
# file_path_3 = f"outputs/GA/Box_DA/GA_seed3_DA.json"
#
# generation_3, individual_3, fitness_3, avg_fitness_3 = read_json_file_GA_pp(file_path_3)
#
#
# print(fitness_1[-1])
# formatted_list = [f"{element:.3f}" for element in individual_1[-1]]
# print(', '.join(formatted_list))
#
# print(fitness_2[-1])
# formatted_list = [f"{element:.3f}" for element in individual_2[-1]]
# print(', '.join(formatted_list))
#
#
# print(fitness_3[-1])
# formatted_list = [f"{element:.3f}" for element in individual_3[-1]]
# print(', '.join(formatted_list))
#
#
# plot_fitness_vs_generation(generation_1, fitness_1, fitness_2, fitness_3, save=True, filename='GA_box_DA.png')
# plot_avg_fitness_vs_generation(generation_1, avg_fitness_1, avg_fitness_2, avg_fitness_3, save=True, filename='avg_box_DA.png')
#
# combined_dataframe = create_combined_dataframe(individual_1[-1])
# plot_outflow_rainfall_benchmark(combined_dataframe, fitness_1[-1], seed=1, save=True)


# # GA Box_DC
# file_path_1 = f"outputs/GA/Box_DC/GA_seed1_DC.json"
#
# generation_1, individual_1, fitness_1, avg_fitness_1 = read_json_file_GA_pp(file_path_1)
#
# file_path_2 = f"outputs/GA/Box_DC/GA_seed2_DC.json"
#
# generation_2, individual_2, fitness_2, avg_fitness_2 = read_json_file_GA_pp(file_path_2)
#
# file_path_3 = f"outputs/GA/Box_DC/GA_seed3_DC.json"
#
# generation_3, individual_3, fitness_3, avg_fitness_3 = read_json_file_GA_pp(file_path_3)
#
# print(fitness_1[-1])
# formatted_list = [f"{element:.3f}" for element in individual_1[-1]]
# print(', '.join(formatted_list))
#
# print(fitness_2[-1])
# formatted_list = [f"{element:.3f}" for element in individual_2[-1]]
# print(', '.join(formatted_list))
#
#
# print(fitness_3[-1])
# formatted_list = [f"{element:.3f}" for element in individual_3[-1]]
# print(', '.join(formatted_list))
#
# # plot_fitness_vs_generation(generation_1, fitness_1, fitness_2, fitness_3, save=True, filename='GA_box_DC.png')
# # plot_avg_fitness_vs_generation(generation_1, avg_fitness_1, avg_fitness_2, avg_fitness_3, save=True, filename='avg_box_DC.png')
#
# combined_dataframe = create_combined_dataframe(individual_1[-1], box_da=False)
# plot_outflow_rainfall_benchmark(combined_dataframe, fitness_1[-1], seed=1, save=True, filename='GA_box_DC')


# # PSO DA
file_path_1 = f"outputs/PSO/Box_DA/logbook_seed_1.json"

generation_1, individual_1, fitness_1, avg_fitness_1 = read_json_file_PSO_pp(file_path_1)
fitness_1 = best_so_far(fitness_1)

file_path_2 = f"outputs/PSO/Box_DA/logbook_seed_2.json"

generation_2, individual_2, fitness_2, avg_fitness_2 = read_json_file_PSO_pp(file_path_2)
fitness_2 = best_so_far(fitness_2)

file_path_3 = f"outputs/PSO/Box_DA/logbook_seed_3.json"

generation_3, individual_3, fitness_3, avg_fitness_3 = read_json_file_PSO_pp(file_path_3)
fitness_3 = best_so_far(fitness_3)



print(fitness_1[-1])
formatted_list = [f"{element:.3f}" for element in individual_1[-1]]
print(', '.join(formatted_list))

print(fitness_2[-1])
formatted_list = [f"{element:.3f}" for element in individual_2[-1]]
print(', '.join(formatted_list))


print(fitness_3[-1])
formatted_list = [f"{element:.3f}" for element in individual_3[-1]]
print(', '.join(formatted_list))

plot_fitness_vs_generation(generation_1, fitness_1, fitness_2, fitness_3, save=True, filename='PSO_DA_parameters.png')



# plot_avg_fitness_vs_generation(generation_1, avg_fitness_1, avg_fitness_2, avg_fitness_3, save=True, filename='PSO_avg_benchmark.png')
#
combined_dataframe = create_combined_dataframe(individual_1[-1])
plot_outflow_rainfall_benchmark(combined_dataframe, fitness_1[-1], seed=1, save=True, filename='PSO_hid_da')

def best_so_far(arr):
    best = [arr[0]]
    for i in range(1, len(arr)):
        best.append(max(best[-1], arr[i]))
    return best

#
# # # PSO DC
# file_path_1 = f"outputs/PSO/Box_DC/logbook_seed_1.json"
#
# generation_1, individual_1, fitness_1, avg_fitness_1 = read_json_file_PSO_pp(file_path_1)
# fitness_1 = best_so_far(fitness_1)
#
# file_path_2 = f"outputs/PSO/Box_DC/logbook_seed_2.json"
#
# generation_2, individual_2, fitness_2, avg_fitness_2 = read_json_file_PSO_pp(file_path_2)
# fitness_2 = best_so_far(fitness_2)
#
# file_path_3 = f"outputs/PSO/Box_DC/logbook_seed_3.json"
#
# generation_3, individual_3, fitness_3, avg_fitness_3 = read_json_file_PSO_pp(file_path_3)
# fitness_3 = best_so_far(fitness_3)
#
# print(fitness_1[-1])
# formatted_list = [f"{element:.3f}" for element in individual_1[-1]]
# print(', '.join(formatted_list))
#
# print(fitness_2[-1])
# formatted_list = [f"{element:.3f}" for element in individual_2[-1]]
# print(', '.join(formatted_list))
#
#
# print(fitness_3[-1])
# formatted_list = [f"{element:.3f}" for element in individual_3[-1]]
# print(', '.join(formatted_list))
#
#
# plot_fitness_vs_generation(generation_1, fitness_1, fitness_2, fitness_3, save=True, filename='PSO_DC_parameters.png')
#
#
# # plot_avg_fitness_vs_generation(generation_1, avg_fitness_1, avg_fitness_2, avg_fitness_3, save=True, filename='PSO_avg_DC.png')
# #
# combined_dataframe = create_combined_dataframe(individual_1[-1], box_da=False)
# plot_outflow_rainfall_benchmark(combined_dataframe, fitness_1[-1], seed=1, save=True, filename='PSO_hid_dc')

#
# # BO Box_DA
# file_path_1 = f"outputs/BO/Box_DA/bayesian_optimization_seed1_box_da.json"
#
# generation_1, individual_1, fitness_1 = read_json_file_BO_benchmark_1(file_path_1)
#
# file_path_2 = f"outputs/BO/Box_DA/bayesian_optimization_seed2_box_da.json"
#
# generation_2, individual_2, fitness_2 = read_json_file_BO_benchmark_1(file_path_2)
#
# file_path_3 = f"outputs/BO/Box_DA/bayesian_optimization_seed3_box_da.json"
#
# generation_3, individual_3, fitness_3 = read_json_file_BO_benchmark_1(file_path_3)
#
# # plot_fitness_vs_generation(generation_1, fitness_1, fitness_2, fitness_3, save=True, filename='BO_BoxDA_parameters.png')
#
# print(fitness_1[-1])
# print(individual_1[-1])
#
# print(fitness_2[-1])
# print(individual_2[-1])
#
# print(fitness_3[-1])
# print(individual_3[-1])

# combined_dataframe = create_combined_dataframe(individual_1[-1], box_da=True)
# plot_outflow_rainfall_benchmark(combined_dataframe, fitness_1[-1], seed=1, save=True, filename='BO_box_DA_hid_benchmark')


# # BO Box_DC
# file_path_1 = f"outputs/BO/Box_DC/bayesian_optimization_seed1_box_da.json"
#
# generation_1, individual_1, fitness_1 = read_json_file_BO_benchmark_1(file_path_1)
#
# file_path_2 = f"outputs/BO/Box_DC/bayesian_optimization_seed2_box_da.json"
#
# generation_2, individual_2, fitness_2 = read_json_file_BO_benchmark_1(file_path_2)
#
# file_path_3 = f"outputs/BO/Box_DC/bayesian_optimization_seed3_box_da.json"
#
# generation_3, individual_3, fitness_3 = read_json_file_BO_benchmark_1(file_path_3)
#
# plot_fitness_vs_generation(generation_1, fitness_1, fitness_2, fitness_3, save=True, filename='BO_BoxDA_parameters.png')
#
# print(fitness_1[-1])
# print(individual_1[-1])
#
# print(fitness_2[-1])
# print(individual_2[-1])
#
# print(fitness_3[-1])
# print(individual_3[-1])

# combined_dataframe = create_combined_dataframe(individual_1[-1], box_da=False)
# plot_outflow_rainfall_benchmark(combined_dataframe, fitness_1[-1], seed=1, save=True, filename='BO_box_DC_hid_benchmark')

#
# # GA Benchmark
# file_path_1 = f"outputs/GA/Benchmark/2/GA_seed1_benchmark.json"
#
# generation_1, individual_1, fitness_1, avg_fitness_1 = read_json_file_GA_benchmark(file_path_1)
#
# file_path_2 = f"outputs/GA/Benchmark/2/GA_seed2_benchmark.json"
#
# generation_2, individual_2, fitness_2, avg_fitness_2 = read_json_file_GA_benchmark(file_path_2)
#
# file_path_3 = f"outputs/GA/Benchmark/2/GA_seed3_benchmark.json"
#
# generation_3, individual_3, fitness_3, avg_fitness_3 = read_json_file_GA_benchmark(file_path_3)
#
# print(fitness_1[-1])
# formatted_list = [f"{element:.3f}" for element in individual_1[-1]]
# print(', '.join(formatted_list))
#
# print(fitness_2[-1])
# formatted_list = [f"{element:.3f}" for element in individual_2[-1]]
# print(', '.join(formatted_list))
#
#
# print(fitness_3[-1])
# formatted_list = [f"{element:.3f}" for element in individual_3[-1]]
# print(', '.join(formatted_list))

# #
# # plot_fitness_vs_generation(generation_1, fitness_1, fitness_2, fitness_3, save=True, filename='GA_benchmark.png')
# # plot_avg_fitness_vs_generation(generation_1, avg_fitness_1, avg_fitness_2, avg_fitness_3, save=True, filename='GA_avg_benchmark.png')
#
# combined_dataframe = create_combined_dataframe_benchmark(individual_1[-1])
# plot_outflow_rainfall_benchmark(combined_dataframe, fitness_1[-1], seed=1, save=True, filename='GA_hid_benchmark')
#


# # # PSO Benchmark
# file_path_1 = f"outputs/PSO/Benchmark/PSO_seed_1.json"
#
# generation_1, individual_1, fitness_1, avg_fitness_1 = read_json_file_PSO_benchmark(file_path_1)
#
# file_path_2 = f"outputs/PSO/Benchmark/PSO_seed_2.json"
#
# generation_2, individual_2, fitness_2, avg_fitness_2 = read_json_file_PSO_benchmark(file_path_2)
#
# file_path_3 = f"outputs/PSO/Benchmark/PSO_seed_3.json"
#
# generation_3, individual_3, fitness_3, avg_fitness_3 = read_json_file_PSO_benchmark(file_path_3)
#
# print(fitness_1[-1])
# formatted_list = [f"{element:.3f}" for element in individual_1[-1]]
# print(', '.join(formatted_list))
#
# print(fitness_2[-1])
# formatted_list = [f"{element:.3f}" for element in individual_2[-1]]
# print(', '.join(formatted_list))
#
#
# print(fitness_3[-1])
# formatted_list = [f"{element:.3f}" for element in individual_3[-1]]
# print(', '.join(formatted_list))


# plot_fitness_vs_generation(generation_1, fitness_1, fitness_2, fitness_3, save=True, filename='PSO_benchmark_parameters.png')
#
# print(fitness_1[-1])
# print(individual_1[-1])
#
# print(fitness_2[-1])
# print(individual_2[-1])
#
# print(fitness_3[-1])
# print(individual_3[-1])
#
# plot_avg_fitness_vs_generation(generation_1, avg_fitness_1, avg_fitness_2, avg_fitness_3, save=True, filename='PSO_avg_benchmark.png')
#
# combined_dataframe = create_combined_dataframe_benchmark(individual_1[-1])
# plot_outflow_rainfall_benchmark(combined_dataframe, fitness_1[-1], seed=1, save=True, filename='PSO_hid_benchmark')


# # BO Benchmark
# file_path_1 = f"outputs/BO/Benchmark/bayesian_optimization_seed1_box_dc.json"
#
# generation_1, individual_1, fitness_1 = read_json_file_BO_benchmark_1(file_path_1)
#
# file_path_2 = f"outputs/BO/Benchmark/bayesian_optimization_seed2_benchmark.json"
#
# generation_2, individual_2, fitness_2 = read_json_file_BO_benchmark(file_path_2)
#
# file_path_3 = f"outputs/BO/Benchmark/bayesian_optimization_seed3_benchmark.json"
#
# generation_3, individual_3, fitness_3 = read_json_file_BO_benchmark(file_path_3)
#
# # plot_fitness_vs_generation(generation_1, fitness_1, fitness_2, fitness_3, save=True, filename='BO_benchmark_parameters.png')
#
# print(fitness_1[-1])
# print(individual_1[-1])
#
# print(fitness_2[-1])
# print(individual_2[-1])
#
# print(fitness_3[-1])
# print(individual_3[-1])
#
# combined_dataframe = create_combined_dataframe_benchmark(individual_1[-1])
# plot_outflow_rainfall_benchmark(combined_dataframe, fitness_1[-1], seed=1, save=True, filename='BO_hid_benchmark')



