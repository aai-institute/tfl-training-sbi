"""Utility functions for the SIR model. """


import numpy as np

from scipy.integrate import odeint


def eval_sir_model(
    theta: np.array,
    initial_cond: tuple = (999, 1, 0),
    population_size: int = 1_000,
    grid_points: np.array = np.linspace(0, 160, 160),
) -> np.array:
    """Evaluate the SIR model for given number of parameters.

    Args:
        theta (np.array): Input params beta and gamma.
        initial_cond (tuple, optional): Initial cond. S0, I0 and R0. Defaults to
        (999, 1, 0), i.e. (N-1, 1, 0).
        population_size (int, optional): Population size. Defaults to 1_000.
        grid_points (np.array, optional): Grid point, i.e. time. Defaults to np.linspace(0, 160, 160).

    Returns:
        np.array: three dim. array of shape (grid_points.shape[0], 3) where the
        order is SIR.
    """
    # unpack initial conditions and params passed to function
    S0, I0, R0, N = initial_cond[0], initial_cond[1], initial_cond[2], population_size
    beta, gamma = theta[0], theta[1]
    t = grid_points

    # define diff. eq. of SIR model
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # vectorized initial cond.
    y0 = S0, I0, R0

    # integrate the SIR equations over the time / grid points
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    return ret
