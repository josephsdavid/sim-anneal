from anneal_oop import simulated_annealing, animation_sa, history_sa
import math
import random
import plotly.io as pio
pio.renderers.default = "browser"

def example_2D(x,y):
    res = 0.2 + x**2 + y**2 - 0.1*math.cos(6.0*3.1415*x) - 0.1*math.cos(6.0*3.1415*y)
    return res

def example_3D(x,y,z):
    inner = x**2 + y**2 - x*y
    outer = inner**0.75
    res = (4/3)*outer + z
    return res

def cross_in_tray(x, y):
    sines = math.sin(x)*math.sin(y)
    sub = (math.sqrt(x**2+y**2))/math.pi
    floaty = abs(100-sub)
    exp = math.exp(floaty)
    inside = abs(sines*exp+1)**0.1
    return(-0.0001*inside)

def outer_minima(x, y):
    left = math.sin(x) * math.cos(y)
    numer = math.sqrt(x**2 + y**2)
    denom = math.pi
    floaty = abs(1-numer/denom)
    inner = abs(left*math.exp(floaty))
    return -inner

def dropwave(x, y):
    square = x**2 + y**2
    numer = 1 + math.cos(12*math.sqrt(square))
    denom = (0.5*square + 2)
    return(-numer/denom)





simanneal = simulated_annealing(example_3D, 0.7, 0.0001, [1, 1, 1], 50, 1000, lower_bound = 0, upper_bound = 2)
best = simanneal.optimize()
history_sa(simanneal, "3d.png")
best




dropanneal = simulated_annealing(dropwave, 0.9, 0.0001, [2.4, 2.4], 50, 1000,  upper_bound = 3 , lower_bound = -3)
best = dropanneal.optimize()
print(best)

ani = animation_sa(dropanneal, -3, 3, 0.1)
history_sa(dropanneal)

ani.view_3D("drop wave")
ani.animate_2D(anim_path="drop_wave.mp4")
ani.history_3D()

crossanneal = simulated_annealing(cross_in_tray, 0.9, 0.0001, [5,-5], 50, 1000, upper_bound = 10, lower_bound = -10)
best = crossanneal.optimize()
print(best)

history_sa(crossanneal)

ani = animation_sa(crossanneal, -10, 10, 0.2)
ani.view_3D("Cross In Tray Function")
ani.animate_2D(anim_path="cross_in_tray.mp4")
ani.history_3D()


outer_anneal = simulated_annealing(outer_minima, 0.9, 0.0001, [4,-4], 50, 1000, upper_bound = 10, lower_bound = -10)
best = outer_anneal.optimize()
print(best)

history_sa(outer_anneal,"outer.png")

outer_ani = animation_sa(outer_anneal, -10, 10, 0.1)
outer_ani.view_3D("outer minima")
outer_ani.animate_2D(anim_path="outer.mp4")
outer_ani.history_3D()
