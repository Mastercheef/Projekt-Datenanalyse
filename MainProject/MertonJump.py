import numpy as np


def merton_jump_paths(S, T, r, m, v, lam, steps, Npaths, sigma):
    size = (steps, Npaths)
    dt = T / steps
    # poisson- distributed jumps
    jumps = np.random.poisson(lam * dt, size=size)

    poi_rv = np.multiply(jumps,
                         np.random.normal(m, v, size=size)).cumsum(axis=0)
    geo = np.cumsum(((r - sigma ** 2 / 2 - lam * (m + v ** 2 * 0.5)) * dt +
                     sigma * np.sqrt(dt) *
                     np.random.normal(size=size)), axis=0)

    return np.exp(geo + poi_rv) * S, jumps
