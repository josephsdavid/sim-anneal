import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from matplotlib.animation import FuncAnimation
from inspect import signature
import plotly.graph_objects as go



def example_2D(x,y):
    res = 0.2 + x**2 + y**2 - 0.1*math.cos(6.0*3.1415*x) - 0.1*math.cos(6.0*3.1415*y)
    return res

def example_3D(x,y,z):
    res = 4/3 * ((x^2 + y^2 -x*y)**(0.75)) + z
    return res

class objective_fun:
    def __init__(self, fun):
        self.fun = fun
        sig = signature(fun)
        self.n_args = len(sig.parameters)

# o = objective_fun(example_2D)

# o2 = objective_fun(example_3D)

class simulated_annealing(objective_fun):
    def __init__(self, fun, p_start, p_end, initial_try, n_cycles = 50, n_rounds = 50, lower_bound=-1.0, upper_bound=1.0):
        super().__init__(fun)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.t_start = -1.0/math.log(p_start)
        self.t_end = -1.0/math.log(p_end)
        # always assume you are an idiot
        if (len(initial_try) != self.n_args):
            print("Your initial try is wrong, you are a fool!")
            return
        self.tries = [initial_try] # this is a list, we make it a list of lists
        self.n_cycles = n_cycles
        self.n_rounds = n_rounds
        self.cooling_rate = (self.t_end/self.t_start)**(1.0/(self.n_cycles -1.0))
        self.current_state = self.tries[0]
        self.global_state = self.current_state
        self.best_objective = self.fun(*self.current_state)
        self.DeltaE_avg = 0.0
        self.DeltaE = None
        self.results = [self.best_objective]
        self.history = [[*self.current_state, self.best_objective]]
        self.n_accepted = 1.0
        self.temperature = self.t_start
        self.accept = None
        self.round = 0
        self.cycle = 0

    def _generate_new_trial(self): # use _ for methods which wouldnt normally be accessed by user
        self.current_state = [x + random.random() - 0.5 for x in self.current_state]
        self.current_state = [max(min(x, self.upper_bound), self.lower_bound) for x in self.current_state]

    def _check_objective(self):
        # calculate local change in energy
        temp_objective = self.fun(*self.current_state)
        self.history.append([ *self.current_state , temp_objective])
        self.DeltaE = abs(temp_objective - self.best_objective)
        if (temp_objective > self.best_objective):
            if (self.cycle == 0 and self.round == 0):
                self.DeltaE_avg = self.DeltaE
            p_worse = math.exp(-self.DeltaE/(self.DeltaE_avg * self.temperature))
            self.accept =  True if (random.random() < p_worse) else False
        else:
            self.accept = True

    def _update_state(self):
        if self.accept:
            self.global_state = self.current_state
            self.best_objective = self.fun(*self.global_state)
            self.n_accepted += 1
            self.DeltaE_avg = (self.DeltaE_avg * (self.n_accepted - 1.0) + self.DeltaE) / self.n_accepted

    def _record_and_update_temp(self):
        self.tries += [self.global_state]
        self.results += [ self.best_objective ]
        self.temperature = self.cooling_rate * self.temperature

    def optimize(self):
        for i in range(self.n_cycles):
            print('cycle: ' + str(i) + ' Temperature: ' + str(self.temperature))
            self.cycle = i
            for j in range(self.n_rounds):
                self.round = j
                self._generate_new_trial()
                self._check_objective()
                self._update_state()
            self._record_and_update_temp()
        # update to get the guy nice, get the actual best
        best_try = [*self.global_state, self.best_objective]
        return best_try

# simanneal = simulated_annealing(example_2D, 0.01, 0.001, [1.0, 1.0], 50, 50)
# best = simanneal.optimize()


class animation_sa:
    def __init__(self, sa: simulated_annealing, lower: float, upper: float, step: float):
        self.xy = [[t[0] for t in sa.tries]] + [[t[1] for t in sa.tries]]
        self.z = sa.results
        self.xyz = [*self.xy, self.z]
        self.history = [[h[i] for h in sa.history] for i in range(3)]
        if sa.n_args != 2:
            print("we cannot animate in {} dimensions!".format(sa.n_args))
            return
        xi, yi = (np.arange(lower, upper, step) for i in range(2))
        self.mesh_x, self.mesh_y = np.meshgrid(xi, yi)
        self.mesh_fn = np.zeros(self.mesh_x.shape)
        for i in range(0, self.mesh_fn.shape[0]):
            for j in range(0, self.mesh_fn.shape[0]):
                self.mesh_fn[i,j] = sa.fun(self.mesh_x[i,j], self.mesh_y[i,j])

    def _animate_helper_2D(self, i):
        self.line.set_xdata(self.xy[0][:i])
        self.line.set_ydata(self.xy[1][:i])

    def animate_2D(self, title="Non-Convex Function", anim_path="sa.mp4"):
        fig, ax = plt.subplots()
        contours = plt.contour(self.mesh_x, self.mesh_y, self.mesh_fn)
        plt.clabel(contours, inline=1, fontsize=10)
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        self.line = ax.plot(self.xy[0][0], self.xy[1][0], 'r-o')[0]
        anim = FuncAnimation(
            fig, self._animate_helper_2D, interval=500, frames =len(self.xy[0])
        )
        plt.draw()
        anim.save(anim_path)
        plt.show()

    def view_3D(self, title="Non-Convex Function"):
        surface = go.Surface(z=self.mesh_fn, x=self.mesh_x, y=self.mesh_y, opacity=0.5)
        trace = go.Scatter3d(
            x=self.xyz[0], y=self.xyz[1], z=self.xyz[2], marker = dict(
                size = 12,
                color = -np.arange(0, len(self.xyz[2])), # set color to an array/list of desired values
                colorscale='Bluered'), visible=True
        )
        fig = go.Figure(data = [ trace,surface ])
        fig.update_layout(title=title, autosize=False,
                          width=1000, height=1000,
                          margin=dict(l=65, r=50, b=65, t=90),
        showlegend=False)
        fig.show()

    def history_3D(self, title="Search History"):
        surface = go.Surface(z=self.mesh_fn, x=self.mesh_x, y=self.mesh_y, opacity=0.5)
        trace = go.Scatter3d(
            x=self.history[0], y=self.history[1], z=self.history[2], mode='markers',marker = dict(
                size = 1,
                color = -np.arange(0, len(self.history[2])), # set color to an array/list of desired values
                colorscale='Bluered')
        )
        fig = go.Figure(data = [ trace,surface ])
        fig.update_layout(title=title, autosize=False,
                          width=1000, height=1000,
                          margin=dict(l=65, r=50, b=65, t=90),
        showlegend=False)
        fig.show()


# viz = animation_sa(simanneal, -1.0, 1.0, 0.1)

def history_sa(sa: simulated_annealing, savepath: str='iterations.png'):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.plot(sa.results, 'r.-')
    ax1.legend(['objective_fn'])
    ax2 = fig.add_subplot(212)
    for i in range(len(sa.tries[0])):
        ax2.plot([t[i] for t in sa.tries])
    ax2.legend([str(x) for x in signature(sa.fun).parameters.keys()])
    plt.savefig(savepath)
    plt.show()




# viz.animate()
