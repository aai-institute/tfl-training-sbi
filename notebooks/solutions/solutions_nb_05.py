"""
Collection of solutions to the exercises provided in notebook 5 - Evaluation of SBI.
"""

# Solution task 1.
theta_test = prior.sample((num_sbc_simulations,))
x_test = simulator(theta_test, simulator_scale=0.05)

ranks, dap_samples = run_sbc(
    theta_test, x_test, posterior, num_posterior_samples=num_posterior_samples
)
fig, ax = sbc_rank_plot(ranks, num_posterior_samples)
