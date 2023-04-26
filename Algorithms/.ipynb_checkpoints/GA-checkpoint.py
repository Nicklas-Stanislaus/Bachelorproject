import numpy as np
### Implemented from describitons in "Practical Genetic Algorithms" 
def GA(fit_func, lims, dims, n_pop = 10, iterations = 10, x_rate = 0.5, mutation_rate = 0.2, visualize = False):
    # Fitness function parameters
    lims = [(lims[0], lims[1], dim) for dim in range(dims)]

    #Derived parameters
    n_keep = int(np.ceil(n_pop * x_rate)) # number of chromosomes to keep
    total_mutations = int(np.ceil(dims * (n_pop - 1) * mutation_rate)) # Exclude 1 chromosome, the best, due to elitism

    # Create inital population of chromosomes
    chromosomes = np.zeros(shape = (n_pop, dims))
    for lim_low, lim_up, dim in lims: 
        chromosomes[:,dim] = np.random.uniform(lim_low, lim_up, n_pop) 

    # Compute cumulative probabilities using ranked weighting
    ranked_weights = []
    sum_n = sum([n for n in range(1, n_keep + 1)])
    p_cumulative = 0
    for i in range(1, n_keep + 1):
        p = (n_keep + 1 - i) / sum_n
        p_cumulative += p 
        ranked_weights.append(p_cumulative)

    # for visualizing data 
    best_points = []

    for i in range(iterations): 
        # Sort by fittest chromosomes in descending order (smallest value ranked 1)
        chromosomes = np.array(sorted(chromosomes, key = lambda x: fit_func(x)))

        # pair chromosomes and mate them to produce ofspring until population restored. 
        offspring_created = 0
        n_replace = n_pop - n_keep
        while offspring_created < n_replace: 

            # select parents among n_keep best chromosomes according to ranked probabilities. If both indexes are the same, redraw
            index_p1, index_p2 = None, None
            while index_p1 == index_p2: 
                index_p1 = next(x[0] for x in enumerate(ranked_weights) if x[1] > np.random.rand())
                index_p2 = next(x[0] for x in enumerate(ranked_weights) if x[1] > np.random.rand())

            # find a randomly selected crossover point
            cp = int(np.ceil(np.random.rand() * dims) - 1)

            # add first offspring - Method for creating offspring uses a combination of extrapolation and crossover 
            beta = np.random.rand()
            p_new1 = chromosomes[index_p1, cp] - beta * (chromosomes[index_p1, cp] - chromosomes[index_p2, cp])
            offspring_index = n_keep + offspring_created
            chromosomes[offspring_index, ] = np.concatenate((chromosomes[index_p1, :cp], p_new1 ,chromosomes[index_p2, cp + 1:]), axis = None)
            offspring_created += 1

            # If no more offspring needed break loop, else add second offspring
            if offspring_created == n_replace: break

            # NOTICE: parents swap sides
            p_new2 = chromosomes[index_p2, cp] + beta * (chromosomes[index_p1, cp] - chromosomes[index_p2, cp])
            offspring_index = n_keep + offspring_created
            chromosomes[offspring_index, ] = np.concatenate((chromosomes[index_p2, :cp], p_new2 ,chromosomes[index_p1, cp + 1:]), axis = None)
            offspring_created += 1

        # Create mutatations in range allowed for the given mutated variable
        for mutation in range(total_mutations):
            row = np.random.randint(low = 1, high = n_pop)
            column = np.random.randint(dims)
            chromosomes[row, column] = np.random.uniform(low = lims[column][0], high = lims[column][1])
        
        if visualize:
            best_points.append(chromosomes[0,]) #data for visualization 
    
    if visualize: 
        return best_points
    else: 
        return chromosomes[0,]