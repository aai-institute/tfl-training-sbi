"""Several functions that ease the work with simulated data. """

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, atleast_2d
from torch.utils.data import Dataset
from torchvision import transforms


class SIRSimulation(Dataset):
    """Dataset that imitates the simulation of the SIR model."""

    def __init__(
        self,
        data_theta: Tensor,
        data_x: Tensor,
        simulator_lag: float = 0.1,
        prior: torch.distributions.Distribution = None,
        transformations: transforms.Compose = None,
    ):
        """Simulate from the SIR model.

        In order to save time, we pre-generated the data for the tutorial. This
        object then imitates simulating the data by drawing random samples from
        the pre-generated data. In reality, one would sample from the prior and
        evaluate the simulation on the sampled parameters.

        Args:
            data_theta (torch.tensor): Parameters.
            data_x (torch.tensor): Observations.
            simulator_lag (float, optional): Time to sleep to imitate the lag of
            a simulation. Defaults to 0.1.
            prior (torch.distributions.Distribution, optional): Prior. Defaults to
            None.
            transformations (transforms.Compose, optional): Transformations.
        """
        super().__init__()
        self.data_theta = data_theta
        self.data_x = data_x
        self.data_length = data_theta.shape[0]
        self.lag = simulator_lag
        self.prior = prior
        self.transformations = transformations

    def __call__(self, num_samples: int = 1) -> Tuple[Tensor, Tensor]:
        """Sample from the pre-generated data.

        Args:
            num_samples (int, optional): Number of samples to draw. Defaults to
            1.

        Returns:
            tuple: theta, x
        """
        # imitate the lag of the simulator
        time.sleep(self.lag)
        # draw an index at random
        idx = torch.randint(low=0, high=self.data_length, size=(num_samples,))

        # add dim when num_samples == 1
        return self.__getitem__(idx)["theta"], self.__getitem__(idx)["x"]

    def __len__(self) -> int:
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.data_length

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            dict: {"theta": theta, "x": x}
        """
        data = {
            "theta": atleast_2d(self.data_theta[idx]),
            "x": atleast_2d(self.data_x[idx]),
        }

        # apply transformations
        if self.transformations:
            data = self.transformations(data)

        return data


class SIRStdScaler:
    """Standardize the SIR data."""

    def __init__(
        self,
        mean_theta: Tensor = torch.tensor([0.45, 0.13]),
        std_theta: Tensor = torch.tensor([0.24, 0.026]),
        mean_x: Tensor = torch.tensor([40.5]),
        std_x: Tensor = torch.tensor([90.35]),
    ):
        """Standardize the SIR data.

        Args:
            mean_theta (torch.Tensor, optional): Empirical mean of theta.
            std_theta (torch.Tensor, optional): Empirical std of theta.
            mean_x (torch.Tensor, optional): Empirical mean of x.
            std_x (torch.Tensor, optional): Empirical std of x.
        """
        self.mean_theta = mean_theta
        self.std_theta = std_theta
        self.mean_x = mean_x
        self.std_x = std_x

    def __call__(self, batch: Tensor) -> Dict[str, Tensor]:
        """Standardize theta and x.

        Args:
            theta (torch.Tensor): Parameters.
            x (torch.Tensor): Observations.

        Returns:
            dict: {"theta": theta, "obs": x}
        """
        theta, x = batch["theta"], batch["obs"]

        theta = (theta - self.mean_theta) / self.std_theta
        x = (x - self.mean_x) / self.std_x

        return {"theta": theta, "obs": x}

    def rescale(self, batch: Tensor) -> Dict[str, Tensor]:
        """Rescale theta and x.

        Args:
            theta (torch.Tensor): Parameters.
            x (torch.Tensor): Observations.

        Returns:
            dict: {"theta": theta, "obs": x}
        """
        theta, x = batch["theta"], batch["obs"]

        theta = theta * self.std_theta + self.mean_theta
        x = x * self.std_x + self.mean_x

        return {"theta": theta, "obs": x}


def load_sir_data(
    base_path: str,
    file_name_thetas: str = "sir_thetas.npy",
    file_name_x: str = "sir_x_obs.npy",
) -> Tuple[Tensor, Tensor]:
    """Load the pre-generated data.

    Args:
        base_path (str): Path to the data.
        fie_name_thetas (str, optional): Name of the file containing thetas.
        file_name_x (str, optional): Name of the file containing the observations.

    Returns:
        tuple: theta, x
    """

    theta = np.load(os.path.join(base_path, file_name_thetas))
    x = np.load(os.path.join(base_path, file_name_x))

    return torch.from_numpy(theta), torch.from_numpy(x)
