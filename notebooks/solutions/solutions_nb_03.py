"""
Collection of solutions to the exercises provided in notebook 3 - Neural Density
Estimation.
"""


# Solution task 1.
# define a torch DatLoader
# preprocessing of data, e.g. standardization happens in the Dataset, i.e. simulator
sir_dataloader = DataLoader(simulator, batch_size=1024, shuffle=True, drop_last=True)


# Solution task 2.
# initialize the mixture density network, parameterizing a
# `num_components`-component mixture of Gaussians with `features`
# features, i.e. variables
mixture_density_net = mdn.MultivariateGaussianMDN(
    features=2,
    hidden_net=nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
    ),
    num_components=4,
    hidden_features=16,
)


# Solution task 3.
num_samples = 50_000
samples = []

# get the posterior, i.e. get the mixture components
weights, means, variances = mixture_density_net.get_mixture_components(x_o.reshape(1, -1))

for _ in trange(num_samples):
    # sample form mixture of Gaussians
    sample = mdn.mog_sample(weights, means, variances)
    samples.append(sample)

samples = torch.cat(samples).detach()
