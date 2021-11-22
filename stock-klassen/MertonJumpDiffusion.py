import matplotlib.pyplot as plt
import numpy as np

"""
Source https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python
stockprice = 100  # current stock price
timeMaturity = 1  # time to maturity
riskFreeRate = 0.02  # risk free rate
meanJumpSize = 0  # mean of jump size
stdJump = 0.3  # standard deviation of jump
jumpIntensity = 1  # intensity of jump i.e. number of jumps per annum
steps = 1000  # time steps
nPaths = 3  # number of paths to simulate
sigmaStdDeviation = 0.2  # annaul standard deviation , for weiner process

j = merton_jump_paths(S, T, r, sigma, lam, m, v, steps, Npaths)
plt.style.use('ggplot')
plt.plot(j)
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.title('Jump Diffusion Process')
"""


def merton_jump_paths(stockprice,
                      timeMaturity,
                      riskFreeRate,
                      sigmaStdDeviation,
                      jumpIntensity,
                      meanJumpSize,
                      stdJump,
                      steps,
                      nPaths):

    size = (steps, nPaths)
    dt = timeMaturity / steps
    poi_rv = np.multiply(np.random.poisson(jumpIntensity * dt, size=size),
                         np.random.normal(meanJumpSize, stdJump, size=size)).cumsum(axis=0)
    geo = np.cumsum(((riskFreeRate - sigmaStdDeviation ** 2 / 2 - jumpIntensity * (meanJumpSize + stdJump ** 2 * 0.5)) * dt +
                     sigmaStdDeviation * np.sqrt(dt) *
                     np.random.normal(size=size)), axis=0)
    return np.exp(geo + poi_rv) * stockprice


class MertonJumpDiffusion:

    def __init__(self):
        pass
