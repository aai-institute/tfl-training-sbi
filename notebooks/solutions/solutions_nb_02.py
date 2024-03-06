"""
Collection of solutions to the exercises provided in notebook 2 - Conditional
Density Estimation.
"""

# Solution task 1.
dataset = TensorDataset(theta, x)

train_loader = DataLoader(dataset, batch_size=20)

# define a simple neural network to parameterize the conditional density
net = nn.Sequential(
    nn.Linear(1, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 2)
)

opt = Adam(net.parameters(), lr=0.001)

# train the neural network for 100 epochs
for e in trange(100):
    for (
        theta_batch,
        x_batch,
    ) in train_loader:
        opt.zero_grad()
        nn_output = net(theta_batch)
        mean = nn_output[:, 0].unsqueeze(1)
        std = torch.exp(nn_output[:, 1]).unsqueeze(1)
        prob_Gauss = (
            1
            / torch.sqrt(2 * math.pi * std**2)
            * torch.exp(-0.5 / std**2 * (mean - x_batch) ** 2)
        )
        loss = -torch.log(prob_Gauss).sum()
        loss.backward()
        opt.step()


# Solution task 2.
# initialize the mixture density network, parameterizing a
# `num_components`-component mixture of Gaussians with `features`
# features, i.e. variables
num_hidden_units = 20
num_hidden_layers = 2
num_features = 1
mixture_density_net = mdn.MultivariateGaussianMDN(
    features=num_features,
    hidden_net=nn.Sequential(
        nn.Linear(num_features, num_hidden_units),
        nn.ReLU(),
        nn.Linear(num_hidden_units, num_hidden_units),
        nn.ReLU(),
        nn.Linear(num_hidden_units, num_hidden_units),
        nn.ReLU(),
    ),
    num_components=5,
    hidden_features=num_hidden_units,
)


# Solution task 3.
# chose value to condition on & compute the mixture components
x_o = torch.as_tensor([[0.5]])
logits, means, variances = mixture_density_net.get_mixture_components(x_o)
theta_test = torch.linspace(-0.1, 1.0, 100)

log_probs = []

for theta_i in tqdm(theta_test):
    # compute probability of different values of theta for the current parameterization
    log_prob = mdn.mog_log_prob(theta_i, logits, means, variances)
    log_probs.append(log_prob)

probs = torch.stack(log_probs).detach().exp()
