"""
Collection of solutions to the exercises provided in notebook 1 - Introduction to SBI.
"""


import torch

from torch import Tensor
from tqdm.notebook import tnrange
from typing import Callable


# Part 1 solution.
def distance(x: Tensor, x_o: Tensor) -> Tensor:
    """Returns the mean squared error (MSE) between x and x_o.

    Note: the mean is taken over data dimensions, i.e., over the second dimension of x.

    Args:
        x: Simulated data, shape (batch, dim_x)
        x_o: Observed data, (1, dim_x)

    Returns:
        distance: MSEs, (batch, 1)
    """
    assert len(x_o.shape) == len(x.shape) == 2, "x and x_o must be 2D."
    assert x_o.shape[0] == 1, "x_o should have shape (1, dim_x)"
    assert x.shape[-1] == x_o.shape[-1], "x and x_o should have the same last dimension"

    return torch.mean((x - x_o) ** 2, -1)


# Part 2 solution.
def rejection_abc(
    num_simulations: int, sample_and_simulate: Callable, distance: Callable, epsilon: float, x_o: Tensor
) -> Tensor:
    """Returns a tensor of accepted posteriors samples obtained with the rejection ABC algorithm.

    Args:
        num_simulations: simulation budget
        sample_and_simulate: a function that samples a parameter from the prior and 
            simulates the SIR model: takes number of samples (int) as input and returns
            theta and x: theta_i, x_i = sample_and_simulate(num_samples)
        distance: a distance function.
        epsilon: the rejection threshold for the distance between x and x_o.
        x_o: the observed data.
    Returns:
        posterior_samples: the accepted theta, i.e., theta for which d(x, x_o) < epsilon.
        theta: all sampled parameters
        x: all simulated data
    """

    posterior_samples = []
    theta = []
    x = []

    for _ in tnrange(num_simulations):
        theta_i, x_i = sample_and_simulate(1)
        theta.append(theta_i)
        x.append(x_i)

        if distance(x_i, x_o) < epsilon:
            posterior_samples.append(theta_i)

    assert len(posterior_samples) > 0, "No samples accepted, consider increasing N or eps."
    
    print(f"Rejection ABC: {len(posterior_samples)} accepted samples.")

    return torch.cat(posterior_samples), torch.cat(theta), torch.cat(x)


# Part 3 solution.
eps = 120
num_simulations = 50000
posterior_samples, theta, x = rejection_abc(num_simulations, sample_and_simulate, distance, eps, x_o)
