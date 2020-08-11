import numpy as np 
import argparse
import matplotlib.pyplot as plt
import os

# Compute Markov bound
def compute_markov_bound(means, variances, quantile):
	expectation = np.sum(variances + np.power(means, 2))
	return np.divide(expectation, quantile)

# Sample generalized chi_2 random variable (assuming diagonal covariance in MVN)
def sample_generalized_chi_2(means, variances, num_samples):
	Gaussian_samples = np.random.multivariate_normal(means, np.diag(variances), size=num_samples)
	generalized_chi_2_samples = np.sum(np.power(Gaussian_samples, 2), axis=1)
	return generalized_chi_2_samples

# Get Monte Carlo estimate of probability
def get_estimated_probability(samples, quantile):
	estimated_probability = len(samples[samples >= quantile]) / len(samples)
	return estimated_probability

# Define generalized chi_2 random variable
def get_generalized_chi_2_params(gaussian_type, dimension, scaling=1.0):
	if gaussian_type == 'random':
		# Random means and random variances (off-center and non-unit variances)
		means = scaling * np.random.rand(dimension)
		variances = scaling * np.abs(np.random.rand(dimension))
	elif gaussian_type == 'scaled_center':
		means = np.zeros(dimension)
		variances = scaling * np.ones(dimension)
	elif gaussian_type == 'off_center':
		means = np.random.rand(dimension)
		variances = np.ones(dimension)
	else:
		means = np.zeros(dimension)
		variances = np.ones(dimension)
	return means, variances

# BB: Taken from Stack Overflow to align axes in plot
def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

def get_gaussian_plot_string(gaussian_type, scaling=1):
	if gaussian_type == 'random':
		return "random mean, random variance"
	elif gaussian_type == 'scaled_center':
		return "$\mathcal{N}(0, " + "{})$".format(scaling)
	elif gaussian_type == 'off_center':
		return "random mean, $\sigma^2=1$"
	else: 
		return "$\mathcal{N}(0, 1)$"

def generate_probability_bound_plot(quantiles, bounds, samples, estimated_probabilities, gaussian_type, scaling=1):
	bounds = np.array(bounds)
	quantiles = np.array(quantiles)

	truncated_bounds = bounds[np.where(bounds <= 10)]
	truncated_quantiles = quantiles[np.where(bounds <= 10)]

	gaussian_plot_string = get_gaussian_plot_string(gaussian_type, scaling=scaling)

	plt.clf()
	fig, ax1 = plt.subplots()

	ax1_color = 'tab:red'
	ax1.hist(samples, color=ax1_color)
	ax1.set_ylabel('Frequency', color=ax1_color)
	ax1.tick_params(axis='y', labelcolor=ax1_color)

	ax2 = ax1.twinx() 
	ax2_color = 'tab:blue'
	ax2.set_ylabel('Probability', color=ax2_color)  # we already handled the x-label with ax1
	ax2.plot(quantiles, estimated_probabilities, 'o', color=ax2_color, label=r'$\mathbb{P}[X > t]$')
	ax2.plot(truncated_quantiles, truncated_bounds, '+', color=ax2_color, label=r'$\frac{\mathbb{E}[X]}{t}$')
	ax2.tick_params(axis='y', labelcolor=ax2_color)

	align_yaxis(ax1, 0, ax2, 0)

	plt.hlines(y=[0.0, 1.0], xmin=0, xmax=max(samples) + 1, color='black')
	plt.title(r"Generalized $\chi^2$ samples, upper tail probabilities & Markov's Bound" "\n" r"(Gaussians used in sampling: {})".format(gaussian_plot_string))
	plt.legend()
	plt.savefig("../out/vae/other_figures/" + "markov_bound_{}_gaussians".format(gaussian_type), dpi=300)
	plt.clf()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--num_samples', type=int, default=int(1e5), help='number of samples used in Monte Carlo estimation of probability')
	parser.add_argument('--gaussian_type', type=str, default='standard', help='type of Gaussians used to define generalized chi^2')
	parser.add_argument('--gaussian_scaling', type=float, default=1.0, help='scaling of Gaussians used to define generalized chi^2')
	parser.add_argument('--dimension', type=int, default=10, help='dimension of multivariate Gaussian used to define generalized chi^2')
	parser.add_argument('--quantile_upper_bound', type=float, default=10.0, help='largest quantile to compute probability / bound for')
	parser.add_argument('--num_quantiles', type=int, default=10, help='number of quantiles to evaluate on')
	parser.add_argument('--verbose', type=bool, default=False, help='verbose mode (True/False)')
	args = parser.parse_args()

	quantiles = np.linspace(1e-3, args.quantile_upper_bound, args.num_quantiles + 1)

	means, variances = get_generalized_chi_2_params(args.gaussian_type, args.dimension, scaling=args.gaussian_scaling)
	generalized_chi_2_samples = sample_generalized_chi_2(means, variances, args.num_samples)

	estimated_probabilities = []
	bounds = []
	for quantile in quantiles:
		estimated_probabilities.append(get_estimated_probability(generalized_chi_2_samples, quantile))
		bounds.append(compute_markov_bound(means, variances, quantile))
	
	if args.verbose:
		print("Quantiles: {}".format(quantiles))
		print("Estimated probabilities: {}".format(estimated_probabilities))
		print("Bounds: {}".format(bounds))

	generate_probability_bound_plot(quantiles, bounds, generalized_chi_2_samples, estimated_probabilities, args.gaussian_type, scaling=args.gaussian_scaling)

