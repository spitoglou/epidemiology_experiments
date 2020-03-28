from IPython.core.display import Image
# import numpy as np
# import pandas as pd
# import random
# import pprint
# import matplotlib.pyplot as plt

# import math
from modelclass import Model
from pathlib import Path
import imageio

# population
n = 100

# grid
x = 10
y = 10

# no of random contacts in one iteration
contacts = 2

# probability of a contact to become infection
inf_prob = 0.7

# time to recovery
ttr = 14

# initial infected population
infected = 2  # no

parameters = {
    'population': n,
    'grid': {
        'x': x,
        'y': y
    },
    'contacts': {
        'number': contacts,
        'type': 'random'
    },
    'initial_infected': infected,
    'probability_of_infection': inf_prob,
    'time_to_recovery': ttr
}

run = Model(parameters, True, True)
run.add_infected()
# print(run.stats)
# print(run.progression_dict)
# run.plot1()
run.plot2()

iterations = 60
for i in range(iterations):
    run.iterate()
    # pprint.pprint(run.progression_dict)
    # run.plot1()
    run.plot2()
run.plot_progression()


image_path = Path('temp/')
images = list(image_path.glob('*.png'))
image_list = []
for file_name in images:
    image_list.append(imageio.imread(file_name))

imageio.mimwrite('animated_from_images.gif', image_list, loop=1, duration=2)

Image(filename='animated_from_images.gif')
