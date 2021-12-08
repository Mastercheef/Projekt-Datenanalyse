import numpy as np


def merton_jump_paths(S=1.0, T=1, r=0.02, m=0, v=0.03, lam=8, steps=1000, Npaths=1, sigma=0.25):
    ''' The function calculates a path of a merton jump model based on the transferred parameters.
    :param S: current stock price
    :param T: time to maturity
    :param r: risk free rate
    :param m: meean of jump size
    :param v: standard deviation of jump
    :param lam:    intensity of jump i.e. number of jumps per annum
    :param steps:  time steps
    :param Npaths: number of paths to simulate
    :param sigma:  annaul standard deviation , for weiner process
    :return: merton-jump-process [list],signed jumps [list] , contamination [float]
    '''
    size = (steps, Npaths)
    dt = T / steps

    # jump rate (i.e) contamination parameter for IF
    contamin = lam * dt

    # poisson- distributed jumps
    jumps = np.random.poisson(lam * dt, size=size)

    #contamin = len(jumps[jumps>0]) /steps

    poi_rv = np.multiply(jumps,
                         np.random.normal(m, v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 * 0.5)) * dt +
                     sigma * np.sqrt(dt) *
                     np.random.normal(size=size)), axis=0)

    return np.exp(geo + poi_rv) * S, jumps, contamin
