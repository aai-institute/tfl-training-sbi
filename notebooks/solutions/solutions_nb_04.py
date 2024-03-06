"""
Collection of solutions to the exercises provided in notebook 4 - NLE and NPE.
"""

# Solution task 1.
# load the pre-simulated data from disk
data_theta, data_x = load_sir_data(c.data)

# separate the "observed" data from the training data
theta_obs, x_obs = data_theta[0, :], data_x[0, :]
theta_train, x_train = data_theta[1:, :], data_x[1:, :]


# Solution task 2.
prior, _, _ = process_prior(
    [
        LogNormal(loc=torch.tensor([math.log(0.4)]), scale=torch.tensor([0.5])),
        LogNormal(loc=torch.tensor([math.log(0.125)]), scale=torch.tensor([0.2])),
    ]
)


# Solution task 3.
# obtain a posterior approx. via NPE
inference = sbi.inference.SNLE(prior=prior, density_estimator="maf")
density_estimator = inference.append_simulations(theta_train, x_train).train(
    training_batch_size=1024, learning_rate=1e-4, max_num_epochs=200
)
posterior = inference.build_posterior(density_estimator, sample_with="mcmc")


# Solution task 4.
num_samples = 1000
posterior_samples = posterior.sample(
    (num_samples,), x=x_obs, method="slice_np_vectorized", num_chains=10
)
prior_samples = theta_train
_ = sbi.analysis.pairplot(
    [prior_samples[:num_samples], posterior_samples],
    points=theta_obs,
    points_colors="r",
    offdiag="scatter",
    hist_diag=dict(bins="auto", density=True),
)
