import random


def characterize(i: int):
    if i == 0:
        return 'Susceptible'
    elif i == 1:
        return 'Infected'
    elif i == 2:
        return 'Recovered'
    else:
        return 'Unknown'


def proba_threshold(threshold, verbose=0):
    '''Returns True with probability <threshold>'''
    r = random.random()
    if verbose > 0:
        print(r)
    result = True
    if r > threshold:
        result = False
    return result
