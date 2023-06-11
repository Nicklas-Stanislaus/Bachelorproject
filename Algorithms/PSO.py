import numpy as np

def PSO(fit_func, lims, dims, n_particles = 10, w = 0.72, c1 = 1.2, c2 = 1.2, v_max = 1, iterations = 10, visualize = False):
    # implemented from pseudocode in "Metaheurestics" book 
    #### Fitness function parameters
    lims = [(lims[0], lims[1], dim) for dim in range(dims)]

    #### initialization 
    #current position of particles - randomly drawn from uniform distribution in the allowed range for variable
    current_pos = np.zeros(shape = (n_particles, dims))
    for lim_low, lim_up, dim in lims: 
        current_pos[:,dim] = np.random.uniform(lim_low, lim_up, n_particles) 

    #current velocity of particle
    velocity = np.zeros(shape = (n_particles, dims)) 
    for lim_low, lim_up, dim in lims: 
        velocity[:,dim] = np.random.uniform(lim_low, lim_up, n_particles) 

    #Best position observed by particle
    best_pos = np.copy(current_pos) 

    #index of overall best particle position 
    index_best_overall = np.argmax([fit_func(x) for x in current_pos]) 

    # data for visualization
    best_points = []

    ### interation loop
    for it in range(iterations): 
        # Movement - calculating new position 
        r1 = np.random.rand(n_particles, 1)
        r2 = np.random.rand(n_particles, 1)
        
        velocity = w * velocity + c1 * r1 * (best_pos - current_pos) \
                                + c2 * r2 * (best_pos[index_best_overall] - current_pos)

        velocity[velocity > v_max] = v_max 
        velocity[velocity < -v_max] = -v_max

        current_pos = current_pos + velocity

        #Confinement - Keeping the particles within allowed bounds 
        for lim_low, lim_up, dim in lims: 
            pos_column = current_pos[:, dim] 
            vel_column = velocity[:, dim] 

            index_where_larger = pos_column > lim_up 
            index_where_smaller = pos_column < lim_low

            # Positions where outside bounds set to either upper lim or lower lim of bound. Velocity set to zero
            pos_column[index_where_larger] = lim_up
            pos_column[index_where_smaller] = lim_low
            vel_column[index_where_smaller + index_where_larger] = 0

        #Memorization - remember the best positions of each particle. NOTICE: this is for minizization
        for i in range(n_particles): 
            if fit_func(current_pos[i]) < fit_func(best_pos[i]):
                #np.copy?
                best_pos[i] = current_pos[i]
            # Remember the overall best position
            if fit_func(best_pos[i]) < fit_func(best_pos[index_best_overall]):
                index_best_overall = i

        if visualize:
            best_points.append(np.copy(best_pos[index_best_overall])) # data for visualization
    
    if visualize: 
        return best_points
    else: 
        return best_pos[index_best_overall]
