import numpy as np
def SA(fit_func, lims, dims, T = 100, L = 10, a = 0.95, iterations = 10, visualize = False):
    # fitness function parameters 
    lims = [(lims[0], lims[1], dim) for dim in range(dims)]

    # saving the best points for visualization 
    points = []
    best_value = 10**100
    best_point = None

    # Initializing point
    x = np.zeros(dims)
    for lim_low, lim_up, dim in lims:
        x[dim] = np.random.uniform(lim_low, lim_up) 

    # While temperature not at freezing temperature 
    for it in range(iterations):
        # iterate for a given temperature 
        for i in range(L): 
            # Applying random permutation within the search space
            y = np.copy(x)
            for lim_low, lim_up, dim in lims:
                scale = (lim_up - lim_low) / 10 # use np.exp(10/T) maybe? 
                # Add a normal distributed value within the bounds 
                y[dim] += np.random.normal(loc = 0, scale = scale) 

            # Correct the values to keep them within bounds
            y[y > lim_up] = lim_up
            y[y < lim_low] = lim_low

            # Evaluate change in energy 
            delta = fit_func(y) - fit_func(x)
            
            # Save best points
            if fit_func(y) < best_value:
                best_point = y
                best_value = fit_func(y)

            #if energy lower accept new state
            if delta < 0:  
                x = y
            # else accept with given probality
            elif np.random.rand() < np.exp((- delta) / T):
                x = y 
          
        if visualize:
            points.append(best_point) # data for plotting

        # Temperature scheduling
        T = T * a
    
    if visualize: 
        return points
    else: 
        return best_point