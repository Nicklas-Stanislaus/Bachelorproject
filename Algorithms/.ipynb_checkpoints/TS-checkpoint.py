import numpy as np
def TS(fit_func, lims, dims, iterations = 10, m = 4, n_neighbors = 10, h0 = 0.2, visualize = False):
    # fitness function parameters 
    lims = [(lims[0], lims[1], dim) for dim in range(dims)]

    # Initial empty tabu list 
    tabu_list = []

    # Inital random point
    x = np.zeros(dims)
    for lim_low, lim_up, dim in lims:
        x[dim] = np.random.uniform(lim_low, lim_up) 

    # Best_sol value is also aspiration criteria
    best_sol = 10**30
    best_point = x

    # saving the points for visualization 
    points = []

    # Iterating through main algorithm 
    for i in range(iterations):
        # Find neighbors of current point in each crown neighborhood, except in immediate radius of current point (the inner radius)
        neighbors = []
        # h_prev initially set to inner radius to prevent sampling in this area 
        h_prev = h0
        for _ in range(n_neighbors): 
            # h_current calculated according to isovolume partitioning formula
            h_current = (h_prev**dims - h_prev**dims / n_neighbors)**(1 / n_neighbors)

            # Keep drawing in crown neighborhood until a point not covered in tabu or meeting aspiration criteria selected
            point_selected = False
            draws = 0
            while not point_selected: 
                # Draw neighbor from specificed crown neigborhood
                u = np.random.normal(0, 1, dims)
                u = (1 / np.linalg.norm(u)) * u
                neighbor = np.random.uniform(low = h_prev, high = h_prev + h_current) * u + x

                # Compute if point in any tabu area
                tabu = [np.linalg.norm(tabu_point - neighbor) < h0 for tabu_point in tabu_list]

                # Select point if not in any tabu area or if it meets aspiration criteria
                if (not any(tabu)) or fit_func(neighbor) < best_sol:
                    point_selected = True
                    neighbors.append(neighbor)

                draws += 1
                # if no value found after 5 draws stop drawing
                if draws == 5:
                    break

            # set new h_prev 
            h_prev += h_current 

        # compute the best neighbor
        neighbors_fit = [fit_func(neighbor) for neighbor in neighbors]
        index_best_neighbor = neighbors_fit.index(min(neighbors_fit))
        best_neighbor = neighbors[index_best_neighbor]

        # add best neighbor to tabu list and set it as current point
        tabu_list.append(best_neighbor)
        x = best_neighbor

        # remove first value in tabu list if overflow
        if len(tabu_list) > m: 
                tabu_list.pop(0)

        # set new best solution if fitness of x better than previous best
        if fit_func(x) < best_sol:
            best_sol = fit_func(x)
            best_pos = x
        
        if visualize:
            points.append(x) # For visualization
    
    if visualize: 
        return points
    else:
        return best_pos