from math import log, pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, tensor


class MultivariateGaussianMDN(nn.Module):
    """
    For a documented version of this code, see:
    https://github.com/mackelab/pyknos/blob/main/pyknos/mdn/mdn.py
    """

    def __init__(
        self,
        features,
        hidden_net,
        num_components,
        hidden_features,
    ):
        super().__init__()

        self._features = features
        self._num_components = num_components

        self._hidden_net = hidden_net
        self._logits_layer = nn.Linear(hidden_features, num_components)
        self._means_layer = nn.Linear(hidden_features, num_components * features)
        self._unconstrained_diagonal_layer = nn.Linear(
            hidden_features, num_components * features
        )

    def get_mixture_components(self, context):
        h = self._hidden_net(context)

        logits = self._logits_layer(h)
        logits = logits - torch.logsumexp(logits, dim=1).unsqueeze(1)
        means = self._means_layer(h).view(-1, self._num_components, self._features)

        log_variances = self._unconstrained_diagonal_layer(h).view(
            -1, self._num_components, self._features
        )
        variances = torch.exp(log_variances)

        return logits, means, variances

    def sample(self, num_samples: int, context: Tensor):
        """Samples from the mdn givne context.

        Note: samples num_samples samples for each context in the batch.

        Returns:
            Tensor: samples of shape (batch_size, num_samples, features)
        """
        assert context.ndim == 2, "context should have a batch dimension."

        # repeat the context to match the number of samples
        context = context.unsqueeze(1).repeat(1, num_samples, 1)
        # reshape context to single batch dim for get_mixture_components
        with torch.no_grad():
            logits, means, variances = self.get_mixture_components(
                context.view(-1, context.size(-1))
            )
        samples = mog_sample(logits, means, variances)

        # reshape batch to match the input shape
        return samples.view(-1, num_samples, self._features)

    def log_prob(self, theta, context):
        logits, means, variances = self.get_mixture_components(context)
        return mog_log_prob(theta, logits, means, variances)


def mog_log_prob(
    theta: Tensor, logits: Tensor, means: Tensor, variances: Tensor
) -> Tensor:
    """Computes the log probability of theta under the mixture of gaussians."""
    _, _, theta_dim = means.size()
    theta = theta.view(-1, 1, theta_dim)

    log_cov_det = -0.5 * torch.log(torch.prod(variances, dim=2))

    a = logits
    b = -(theta_dim / 2.0) * log(2 * pi)
    c = log_cov_det
    d1 = theta.expand_as(means) - means
    precisions = 1.0 / variances
    exponent = torch.sum(d1 * precisions * d1, dim=2)
    exponent = tensor(-0.5) * exponent

    return torch.logsumexp(a + b + c + exponent, dim=-1)


def mog_sample(logits: Tensor, means: Tensor, variances: Tensor) -> Tensor:
    """Samples from a mixture of gaussians."""

    # normalize the logits to be a valid probability distribution
    probs = F.softmax(logits, dim=-1)

    # choose a component index for each sample
    choices = torch.multinomial(probs, num_samples=1, replacement=True).view(-1)

    # select means and variances for the chosen components index?
    chosen_means = means[range(means.size(0)), choices, :]
    chosen_variances = variances[range(variances.size(0)), choices, :]

    # sample from a standard normal and scale by the chosen variance
    batch_size = means.shape[0]
    standard_normal_sample = torch.randn(batch_size, 1)
    zero_mean_samples = standard_normal_sample * torch.sqrt(chosen_variances)
    samples = chosen_means + zero_mean_samples

    return samples
