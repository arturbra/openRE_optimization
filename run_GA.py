import random
from deap import base, creator, tools
import json
import time


def bounded_mutGaussian(individual, mu, sigmas, indpb, lower_bound, upper_bound):
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            individual[i] += random.gauss(mu, sigmas[i])
            individual[i] = max(lower_bound[i], min(individual[i], upper_bound[i]))
    return individual,


def bounded_mutGaussian_single_gene(individual, mu, sigmas, lower_bound, upper_bound):
    size = len(individual)
    # Choose one gene index to mutate
    gene_to_mutate = random.randint(0, size - 1)

    # Mutate the selected gene
    individual[gene_to_mutate] += random.gauss(mu, sigmas[gene_to_mutate])
    individual[gene_to_mutate] = max(lower_bound[gene_to_mutate],
                                     min(individual[gene_to_mutate], upper_bound[gene_to_mutate]))

    return individual,


def save_generation_stats(population, generation, seed):
    fits = [ind.fitness.values[0] for ind in population]
    min_fit = min(fits)
    max_fit = max(fits)
    avg_fit = sum(fits) / len(fits)

    # Calculate the standard deviation
    sum2 = sum(x * x for x in fits)
    std_dev = abs(sum2 / len(fits) - avg_fit ** 2) ** 0.5

    best_ind = tools.selBest(population, 1, fit_attr='fitness')[0]
    best_fit = best_ind.fitness.values

    # Create a dictionary with the data for this generation
    result_dict = {'generation': generation,
                   'individual': best_ind,
                   'fitness': best_fit,
                   'min_fitness': min_fit,
                   'max_fitness': max_fit,
                   'avg_fitness': avg_fit,
                   'std_dev_fitness': std_dev}

    # Append the current generation's data as a new line to the JSON file
    json_filename = f"GA_seed{seed}_benchmark.json"
    result_json_string = json.dumps(result_dict)
    with open(json_filename, "a") as outfile:
        outfile.write(result_json_string + '\n')

    return min_fit, avg_fit, max_fit, best_ind, best_fit, std_dev


def initialize_toolbox(calibration, benchmark=True):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("thetaR", random.uniform, calibration.thetaR_min, calibration.thetaR_max)
    toolbox.register("thetaS", random.uniform, calibration.thetaS_min, calibration.thetaS_max)
    toolbox.register("alpha", random.uniform, calibration.alpha_min, calibration.alpha_max)
    toolbox.register("n", random.uniform, calibration.n_min, calibration.n_max)
    toolbox.register("Ks", random.uniform, calibration.Ks_min, calibration.Ks_max)
    toolbox.register("psi0", random.uniform, calibration.psi0_min, calibration.psi0_max)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.thetaR, toolbox.thetaS, toolbox.alpha, toolbox.n, toolbox.Ks, toolbox.psi0))
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    if benchmark:
        toolbox.register('evaluate', calibration.evaluate_by_nash_benchmark)

    else:
        toolbox.register('evaluate', calibration.evaluate_by_nash_pp)

    lower_bounds = [calibration.thetaR_min, calibration.thetaS_min, calibration.alpha_min, calibration.n_min,
                    calibration.Ks_min, calibration.psi0_min]
    upper_bounds = [calibration.thetaR_max, calibration.thetaS_max, calibration.alpha_max, calibration.n_max,
                    calibration.Ks_max, calibration.psi0_max]

    toolbox.register("mate",
                     lambda ind1, ind2: tools.cxSimulatedBinaryBounded(ind1, ind2, low=lower_bounds, up=upper_bounds,
                                                                       eta=30))

    sigma_fraction = 0.15  # The fraction of the parameter range to use as the sigma value
    sigmas = [(upper_bounds[i] - lower_bounds[i]) * sigma_fraction for i in range(len(lower_bounds))]

    toolbox.register('mutate', bounded_mutGaussian, mu=0, sigmas=sigmas, indpb=0.25, lower_bound=lower_bounds,
                     upper_bound=upper_bounds)
    toolbox.register('select', tools.selTournament, tournsize=3)

    return toolbox


def water_flow_calibration(toolbox, calib, s):
    print("Calibration started:")
    random.seed(s)
    pop = toolbox.population(n=calib.pop)
    cx_prob, mut_prob, ngen = 0.5, 0.1, calib.gen
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(ngen):
        tic = time.time()
        print("=========================")
        print("Generation {}".format(g))

        # choose the next generation
        offspring = toolbox.select(pop, k=int(calib.pop / 2))

        # clone selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # compute the crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:

                child1[:], child2[:] = toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # evaluate invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        best_ind = tools.selBest(pop, 1)[0]
        pop[:] = offspring[:-1] + [best_ind]

        min_fit, avg_fit, max_fit, best_ind, best_fit, std_dev = save_generation_stats(pop, g, s)

        print(f"Generation {g}:")
        print(f"  Min Fitness: {min_fit}")
        print(f"  Max Fitness: {max_fit}")
        print(f"  Avg Fitness: {avg_fit}")
        print(f"  Std Dev Fitness: {std_dev}")
        print(f"  Best Individual: {best_ind}")
        print(f"  Best Fitness: {best_fit}")

        runtime = time.time() - tic
        time_left = runtime * (ngen - g) / 60
        print('approx time left = %.2f minutes' % (time_left))

    best_ind = tools.selBest(pop, 1)[0]
    print("Done!")
    print("Best_ind:", best_ind, "Best_fit:", best_ind.fitness.values)


