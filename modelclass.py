import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
from helpers import proba_threshold, characterize
from loguru import logger


class Model:

    def __init__(self, parameters, only_infected=True, story=False):
        self.story = story
        self.only_infected = only_infected
        self.n = parameters['population']
        self.num_infected = parameters['initial_infected']
        self.contacts = parameters['contacts']
        self.probability_of_infection = parameters['probability_of_infection']
        self.ttr = parameters['time_to_recovery']
        self.verbose = 0
        # state
        # numpy array of [a , b]
        # a: category (0:Susceptible, 1:Infected, 2:Recovered)
        # b: # of iterations in the same category
        self.state = np.zeros((self.n, 2))
        # stats
        # (#_susceptible, #_infected, #_recovered)
        self.stats = (self.n, 0, 0)
        self.iteration_counter = 0
        self.progression_dict = {}
        self.progression_dict2 = {}
        self.grid_x = parameters['grid']['x']
        self.grid_y = parameters['grid']['y']
        if self.verbose > 0:
            print(self.state)
            print(self.stats)

    def set_verbosity(self, value=1):
        self.verbose = value

    def add_infected(self):
        infected_index = random.sample(list(range(self.n)), self.num_infected)
        if self.verbose > 0:
            print('initial infected', infected_index)
        for inf in infected_index:
            self.state[inf, 0] = 1
        if self.verbose > 1:
            print(self.state)  # initial with infected
        self.update_stats()

    def get_contacts(self, current_index):
        if self.contacts['type'] == 'random':
            indexes = list(range(self.n))
            indexes.pop(current_index)
            contacts_index = random.sample(indexes, self.contacts['number'])
        if self.verbose > 0:
            print('contacts', contacts_index)
        return contacts_index

    def update_stats(self):
        '''counts of final state'''
        cnt_susceptible = 0
        cnt_infected = 0
        cnt_recovered = 0
        for i, item in enumerate(self.state):
            if item[0] == 0:
                cnt_susceptible += 1
            if item[0] == 1:
                cnt_infected += 1
            if item[0] == 2:
                cnt_recovered += 1
        self.stats = (cnt_susceptible, cnt_infected, cnt_recovered)
        self.snapshot()

    def snapshot(self):
        if self.story:
            print('At this point we have {} Susceptible,'
                  '{} Infected and {} Recovered individuals in the population!'.format(
                      self.stats[0], self.stats[1], self.stats[2]))
        self.progression_dict[self.iteration_counter] = self.stats
        self.progression_dict2[self.iteration_counter] = {
            'Susceptible': self.stats[0],
            'Infected': self.stats[1],
            'Recovered': self.stats[2],
        }

    def iterate(self):
        self.iteration_counter += 1
        logger.info(self.iteration_counter)
        if self.story:
            print('We are now in Iteration no {}'.format(self.iteration_counter))
        # create an immutable copy of the initial array
        # in order to compare later and increase days in the same condition
        immutable = self.state.copy()
        after = self.state.copy()

        # one iteration
        for i, item in enumerate(self.state):
            state_per = item[0]
            if state_per != 1 and self.only_infected:
                continue
            if self.verbose > 0:
                print(i, item, characterize(item[0]))

            if self.story:
                print(
                    'Let\'s see what individual no {} will do in this iteration'.format(i))

            # get [contacts] random persons after removing self
            contacts_index = self.get_contacts(i)

            # process contacts
            for con in contacts_index:
                state_con = after[con][0]
                if self.story:
                    print('He meets with {} who is {}'.format(
                        con, characterize(state_con)))
                if self.verbose > 0:
                    print('contact with {} who is {}'.format(
                        con, characterize(state_con)))

                new_state_per = state_per
                new_state_con = state_con
                if state_per == 1 and state_con == 0:
                    if proba_threshold(self.probability_of_infection):
                        new_state_con = 1
                        if self.story:
                            print('{} is unlucky and {} infects him'.format(con, i))
                    else:
                        if self.story:
                            print(
                                '{} is lucky and {} doen not infects him'.format(con, i))
                if state_per == 0 and state_con == 1:
                    if proba_threshold(self.probability_of_infection):
                        new_state_per = 1
                        if self.story:
                            print('{} is unlucky and {} infects him'.format(i, con))
                    else:
                        if self.story:
                            print(
                                '{} is lucky and {} doen not infects him'.format(i, con))

                after[i][0] = new_state_per
                after[con][0] = new_state_con
                if self.verbose > 1:
                    print(after)

        # compare each item with immutable copy
        # if change of state set iteration counter back to zero
        # else raise by one
        for i, item in enumerate(after):
            # print(i, item[0], immutable[i][0])
            if item[0] == immutable[i][0]:
                after[i][1] += 1
            else:
                after[i][1] = 0

            # recovery
            if item[0] == 1 and item[1] >= self.ttr:
                if self.story:
                    print('{} recovers after {} days of infection'.format(i, self.ttr))
                after[i][0] = 2
                after[i][1] = 0

        if self.verbose > 1:
            print(after)
        self.state = after.copy()
        self.update_stats()

    def plot1(self):
        cmap = mpl.cm.get_cmap('brg')  # colormap used
        x = list(range(len(self.state)))
        x = list(map(lambda i: i % 10, x))
        y = list(range(int(len(self.state) / 10))) * 10
        x.sort()
        if self.verbose > 0:
            print(x)
            print(y)
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c=self.state[:, 0], cmap=cmap)
        plt.show()
        plt.close()

    def plot2(self):
        cmap = mpl.colors.ListedColormap(['blue', 'orange', 'green'])
        bounds = [-0.5, .5, 1.5, 2.5]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        test = self.state[:, 0]
        arr = np.reshape(test, (self.grid_x, self.grid_y)).T
        # current['state'].shape
        x = np.arange(0, self.grid_x + 1)
        y = np.arange(0, self.grid_y + 1)
        plt.figure(figsize=(8, 6))
        plt.title('Iteration {}'.format(self.iteration_counter))
        plt.pcolormesh(x, y, arr, cmap=cmap, norm=norm)
        plt.colorbar()
        plt.savefig('temp/{:03d}.png'.format(self.iteration_counter))
        # plt.show()
        plt.close()

    def plot_progression(self):
        df = pd.DataFrame.from_dict(self.progression_dict2, orient='index')
        df.plot()
        plt.show()
        plt.close()
