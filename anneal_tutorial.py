import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from matplotlib.animation import FuncAnimation

# this code is inspired by an excellent blog post. Please refer to
# https://apmonitor.com/me575/index.php/Main/SimulatedAnnealing

# define objective function
# in the case of machine learning this would be a loss function
def objective_function(x,y):
    res = 0.2 + x**2 + y**2 - 0.1*math.cos(6.0*3.1415*x) - 0.1*math.cos(6.0*3.1415*y)
    return(res)

"""
simulated annealing: optimize a function through simulated annealing!
--------------------------------

inputs:
    objective: function, the objective function

    p_start: the probability of  accepting a worse solution at the start

    p_end: the probability of accepting a worse solution at the end

    initial_try: starting points, list

    n_cycles: the number of tries. Defaults to 50

    n_rounds: number of tries per cycle. Defaults to 50

outputs:
    best_try: list

    cycle_results: list of best tries per cycle
"""
def simulated_annealing(objective, p_start, p_end, initial_try, n_cycles=50, n_rounds=50):
    # initial and final temperatures
    t_initial = -1.0/math.log(p_start)
    t_final = -1.0/math.log(p_end)
    # temperature reduction every cycle
    frac = (t_final/t_initial)**(1.0/(n_cycles-1.0))
    # change in energy
    DeltaE_avg = 0.0
    # number of accepted solutions:
    na = 0.0 # one because we are accepting the first solution
    # list of tries, per cycle:
    tries = [initial_try]
    # objective function results
    results = []
    # initialize
    xi = tries[0][0]
    yi = tries[0][1]
    # calculate current result with the initial solution
    x_current = xi
    y_current = yi
    f_current = objective(xi, yi)
    results.append(f_current)
    na = na + 1.0
    # current temperature
    t = t_initial
    for i in range(n_cycles):
        print('cycle: ' + str(i) + ' Temperature: ' + str(t))
        for j in range(n_rounds):
            # generate random new trial points
            xi, yi = (w + random.random() - 0.5 for w in [x_current, y_current])
            # clip the trial points
            xi, yi = (max(min(w, 1.0), -1.0) for w in [xi, yi])
            # calculate local change of energy
            DeltaE = abs(objective(xi, yi) - f_current)
            if (objective(xi, yi) > f_current):
                # update average energy change if we found a worse solution on
                # fisrst iteration
                if(i == 0 and j == 0):
                    DeltaE_avg = DeltaE
                # probability of accepting the worse solution
                p = math.exp(-DeltaE/(DeltaE_avg * t))
                # determine whether to accept worse point
                if (random.random() < p):
                    accept = True
                else:
                    accept = False
            else:
                # if objective function is better, automatically accept
                accept = True
            if (accept == True):
                # update current best solution
                x_current = xi
                y_current = yi
                f_current = objective(x_current, y_current)
                # update number of accepted solutions
                na = na + 1.0
                # update average energy change
                DeltaE_avg = (DeltaE_avg * (na-1.0) + DeltaE) / na
        # record best values at end of every cycle
        tries = tries + [[x_current, y_current]]
        results.append(f_current)
        # lower_bound the temperature for next cycle
        t = frac*t
    # best solution
    best_try = [x_current, y_current, f_current]
    # cycle results, a list of lists where each list is a value
    # there are definitely better ways to do this
    cycle_results = [[t[0] for t in tries]] + [[t[1] for t in tries]] + [results]
    return(best_try, cycle_results)

best, cyc = simulated_annealing(objective_function, 0.7, 0.001, [0.5, 0.5], 50, 50)

print(best)
# make sure initial try is good
print([c[0] for c in cyc])

# visualization
# create the meshes
# function to create a mesh. This is used not for the simulated annealing, but
# for the visualization. Do not worry too much about it, I will point out the
# code you actually need to do simulated annealing on your own
def mesh(lower_bound, upper_bound, step, objective = objective_function):
    xi = np.arange(lower_bound, upper_bound, step)
    yi = np.arange(lower_bound, upper_bound, step)
    # create mesh coords
    mesh_x, mesh_y = np.meshgrid(xi,yi)
    # create mesh of objective
    # first allocate the space
    mesh_objective = np.zeros(mesh_x.shape)
    # next we populate
    for i in range(0, mesh_x.shape[0]):
        for j in range(0, mesh_x.shape[1]):
            mesh_objective[i,j] = objective(mesh_x[i,j], mesh_y[i,j])
    return(mesh_x, mesh_y, mesh_objective)
xmesh, ymesh, fmesh = mesh(-1.0, 1.0, 0.01, objective_function)

# code for animation
def animate(i):
    line.set_xdata(cyc[0][0:i])
    line.set_ydata(cyc[1][0:i])

fig, ax = plt.subplots()
# Specify contour lines
#lines = range(2,52,2)
# Plot contours
CS = plt.contour(xmesh, ymesh, fmesh)#,lines)
# Label contours
plt.clabel(CS, inline=1, fontsize=10)
# Add some text to the plot
plt.title('Non-Convex Function')
plt.xlabel('x')
plt.ylabel('y')

line = ax.plot(cyc[0][0], cyc[1][0], 'r-o')[0]
anim = FuncAnimation(
    fig, animate, interval = 500, frames = 50
)
plt.draw()
anim.save('sa.mp4')

plt.show()

# show our progress
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(cyc[2],'r.-')
ax1.legend(['Objective'])
ax2 = fig.add_subplot(212)
ax2.plot(cyc[0],'b.-')
ax2.plot(cyc[1],'g--')
ax2.legend(['x','y'])
plt.savefig('iterations.png')
plt.show()

"""
Things to think about
------------------------

How can we update this for any number of dimensions?

This is a situation where OOP makes a lot of sense, how can we improve this code
using that?

What other optimization functions can we try/discuss (metropolis hastings?)?
"""
