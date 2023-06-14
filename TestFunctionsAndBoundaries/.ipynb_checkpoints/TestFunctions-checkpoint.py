import numpy as np

# def area [-5.12, 5.12] with optimum at (0,0,0,..,0) with value 0
def Sphere(v):
    y = sum([v[i]**2 for i in range(len(v))])
    return y
Sphere_bounds = [-5.12, 5.12]
Sphere_opt = 0

# Def space [-100, 100] with optimum at (0,0,0,..,0) with value 0
def Elliptic(v):
    y = sum([10**(6*((i-1)/(len(v))))*v[i]**2 for i in range(len(v))])
    return y
Elliptic_bounds = [-100, 100]
Elliptic_opt = 0

# def area [-5.12, 5.12] with optimum at (0,0,0,..,0) with value 0
def Rastrigin(v): 
    y = 10 * len(v) + sum([v[i]**2 - 10*np.cos(2*np.pi*v[i]) for i in range(len(v))])
    return y
Rastrigin_bounds = [-5.12, 5.12]
Rastrigin_opt = 0

# def space [-32.7, 32.7] with optimum at (0,0,0,..,0) with value 0
def Ackley(v):
    exp_inner1 = -0.2*np.sqrt((1/len(v))*sum([v[i]**2 for i in range(len(v))]))
    exp_inner2 = (1/len(v))*sum([np.cos(2*np.pi*v[i]) for i in range(len(v))])
    y = -20*np.exp(exp_inner1) - np.exp(exp_inner2) + 20 + np.e
    return y
Ackley_bounds = [-32.7, 32.7]
Ackley_opt = 0 

# def space [-100, 100] with optimum at (0,0,0,..,0) with value 0
def Schwefel_1_2(v):
    y = sum([sum([v[j] for j in range(i)])**2 for i in range(len(v))])
    return y
Schwefel_1_2_bounds = [-100, 100]
Schwefel_1_2_opt = 0 

# def space [-500, 500] with optimum at (420.968746359982025,420.968746359982025,..,420.968746359982025) with value 0
def Schwefel(v):
    y = 418.9828872724337*len(v) - sum([v[i] * np.sin(np.sqrt(abs(v[i]))) for i in range(len(v))])
    return y
Schwefel_bounds = [-500, 500]
Schwefel_opt = 0

# def space [-5, 10] with optimum at (1,1,1,...,1) with value 0
def Rosenbrock(v):
    y = sum([100*(v[i+1] - v[i]**2)**2 + (1 - v[i])**2 for i in range(len(v) - 1)])
    return y 
Rosenbrock_bounds = [-5, 10]
Rosenbrock_opt = 0 

# def space [-600, 600] with optimum at (0,0,0..,0) with value 0 
def Griewank(v):
    y = sum([(v[i]**2)/4000 for i in range(len(v))]) - np.prod([np.cos(v[i]/np.sqrt(i + 1)) for i in range(len(v))]) + 1 
    return y
Griewank_bounds = [-600, 600]
Griewank_opt = 0

# def space [-1, 1] with optimum at (0,0,0..,0) with value 0 
def Sum_of_different_powers(v): 
    y = sum([abs(v[i])**(i+2) for i in range(len(v))])
    return y 
Sum_of_different_powers_bounds = [-1, 1]
Sum_of_different_powers_opt = 0

# def space [-10, 10] with optimum at (0,0,0..,0) with value 0 
def Sum_squares(v):
    y = sum([(i+1)*v[i]**2 for i in range(len(v))])
    return y
Sum_squares_bounds = [-10, 10]
Sum_squares_opt = 0