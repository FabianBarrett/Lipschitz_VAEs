import numpy as np 
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# Get generalized chi_2 samples from Gaussian samples
def gaussians_to_generalized_chi_2s(gaussian_samples):
	return np.sum(np.power(gaussian_samples, 2), axis=1)

# Compute estimated probabilities for shifted and scaled generalized chi_2
def get_estimated_shifted_scaled_probabilities(unreduced_gaussian_samples, means, variances, original_quantile):
	if np.sqrt(original_quantile) -  np.linalg.norm(means, ord=2) < 0:
		return np.inf, np.inf, np.inf
	else:
		shifted_quantile = np.power(np.sqrt(original_quantile) -  np.linalg.norm(means, ord=2), 2)
		shifted_scaled_quantile = shifted_quantile / max(variances)
		shifted_samples = np.zeros(unreduced_gaussian_samples.shape)
		for index in range(len(means)):
			shifted_samples[:, index] = unreduced_gaussian_samples[:, index] - means[index]
		# Scaled chi_2 samples are generated using N(0, sigma_2) for some sigma_2
		scaled_chi_2_samples = gaussians_to_generalized_chi_2s(shifted_samples)
		estimated_shifted_probability = get_estimated_probability(scaled_chi_2_samples, shifted_quantile)
		shifted_scaled_samples = np.zeros(unreduced_gaussian_samples.shape)
		for index in range(len(variances)):
			shifted_scaled_samples[:, index] = shifted_samples[:, index] / np.sqrt(variances[index])
		chi_2_samples = gaussians_to_generalized_chi_2s(shifted_scaled_samples)
		estimated_shifted_scaled_probability = get_estimated_probability(chi_2_samples, shifted_scaled_quantile)
		return estimated_shifted_probability, estimated_shifted_scaled_probability, shifted_scaled_quantile

# Compute tail bound for standard chi_2 random variable (as in https://www.math.uni.wroc.pl/~pms/files/30.2/Article/30.2.10.pdf)
def compute_chi_2_bound(df, quantile): 
	if quantile > (df - 2) and quantile < np.inf:
		coefficient = (1.0 / np.sqrt(np.pi)) * (quantile / (quantile - df + 2))
		exponential_term = np.exp(-0.5 * (quantile - df - (df - 2) * np.log(quantile / df) + np.log(df)))
		return coefficient * exponential_term
	else:
		return np.inf

# Compute Markov bound
def compute_markov_bound(means, variances, quantile):
	expectation = np.sum(variances + np.power(means, 2))
	return np.divide(expectation, quantile)

# Sample generalized chi_2 random variable (assuming diagonal covariance in MVN)
def sample_unreduced_gaussians(means, variances, num_samples):
	unreduced_gaussian_samples = np.random.multivariate_normal(means, np.diag(variances), size=num_samples)
	return unreduced_gaussian_samples

# Get Monte Carlo estimate of probability
def get_estimated_probability(samples, quantile):
	estimated_probability = len(samples[samples >= quantile]) / len(samples)
	return estimated_probability

# Define generalized chi_2 random variable
def get_generalized_chi_2_params(gaussian_type, dimension, scaling=1):
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
		return "random means in (-1, 1), random variances in [0, 1)"
	elif gaussian_type == 'scaled_center':
		return "$\mathcal{N}(0, " + "{})$".format(scaling)
	elif gaussian_type == 'off_center':
		return "random means in (-1, 1), $\sigma^2=1$"
	else: 
		return "$\mathcal{N}(0, 1)$"

def generate_markov_probability_bound_plot(quantiles, bounds, samples, estimated_probabilities, gaussian_type, scaling=1.0):

	colors = [color for color in mcolors.TABLEAU_COLORS][:3]

	bounds = np.array(bounds)
	quantiles = np.array(quantiles)

	truncated_bounds = bounds[np.where(bounds <= 10)]
	truncated_quantiles = quantiles[np.where(bounds <= 10)]

	gaussian_plot_string = get_gaussian_plot_string(gaussian_type, scaling=scaling)

	plt.clf()
	fig, ax1 = plt.subplots()

	ax1.hist(samples, bins=30, color=colors[0])
	ax1.set_ylabel('Frequency', color=colors[0])
	ax1.tick_params(axis='y', labelcolor=colors[0])

	ax2 = ax1.twinx() 
	ax2.set_ylabel('Probability', color=colors[1])  # we already handled the x-label with ax1
	ax2.plot(quantiles, estimated_probabilities, 'o', color=colors[1], label=r'$\mathbb{P}[X > t]$')
	ax2.plot(truncated_quantiles, truncated_bounds, '+', color=colors[2], label=r'$\frac{\mathbb{E}[X]}{t}$')
	ax2.tick_params(axis='y', labelcolor=colors[1])

	align_yaxis(ax1, 0, ax2, 0)

	plt.hlines(y=[0.0, 1.0], xmin=0, xmax=max(samples) + 1, color='black')
	plt.title(r"Generalized $\chi^2$ samples, upper tail probabilities & Markov's Inequality" "\n" r"({})".format(gaussian_plot_string))
	plt.legend()
	plt.savefig(os.getcwd() + "/out/vae/other_figures/bound_tightness/" + "markov_bound_{}_gaussians".format(gaussian_type), dpi=300)
	plt.clf()

def generate_chi_2_probability_bound_plot(quantiles, chi_2_bounds, markov_bounds, samples, estimated_probabilities, gaussian_type, scaling=1.0, df=None, combined=False):

	colors = [color for color in mcolors.TABLEAU_COLORS][:4]
	chi_2_bounds = np.array(chi_2_bounds)
	markov_bounds = np.array(markov_bounds)
	quantiles = np.array(quantiles)

	truncated_chi_2_quantiles = quantiles[np.where(chi_2_bounds <= 10)]
	truncated_markov_quantiles = quantiles[np.where(markov_bounds <= 10)]
	truncated_chi_2_bounds = chi_2_bounds[np.where(chi_2_bounds <= 10)]
	truncated_markov_bounds = markov_bounds[np.where(markov_bounds <= 10)]

	gaussian_plot_string = get_gaussian_plot_string(gaussian_type, scaling=scaling)

	plt.clf()
	fig, ax1 = plt.subplots()

	ax1.hist(samples, bins=30, color=colors[0])
	ax1.set_ylabel('Frequency', color=colors[0])
	ax1.tick_params(axis='y', labelcolor=colors[0])

	ax2 = ax1.twinx() 
	ax2.set_ylabel('Probability', color=colors[1])  # we already handled the x-label with ax1
	ax2.plot(quantiles, estimated_probabilities, 'o', color=colors[1], label=r'$\mathbb{P}[X > t]$')
	ax2.plot(truncated_chi_2_quantiles, truncated_chi_2_bounds, '+', color=colors[3], label=r'$\chi^2$ tail bound')
	if combined:
		ax2.plot(truncated_markov_quantiles, truncated_markov_bounds, '+', color=colors[2], label="Markov's Inequality")
	ax2.tick_params(axis='y', labelcolor=colors[1])

	align_yaxis(ax1, 0, ax2, 0)

	plt.hlines(y=[0.0, 1.0], xmin=0, xmax=max(samples) + 1, color='black')
	plt.legend()
	if combined:
		plt.title(r"Generalized $\chi^2$ samples, upper tail probabilities & tail bounds" "\n" r"({} Gaussians, {})".format(df, gaussian_plot_string))
		plt.savefig(os.getcwd() + "/out/vae/other_figures/bound_tightness/" + "combined_bounds_{}_gaussians".format(gaussian_type), dpi=300)
	else:
		plt.title(r"Generalized $\chi^2$ samples, upper tail probabilities & $\chi^2$ tail bound" "\n" r"({} Gaussians, {})".format(df, gaussian_plot_string))
		plt.savefig(os.getcwd() + "/out/vae/other_figures/bound_tightness/" + "chi_2_bound_{}_gaussians".format(gaussian_type), dpi=300)
	plt.clf()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('--num_samples', type=int, default=int(1e5), help='number of samples used in Monte Carlo estimation of probability')
	parser.add_argument('--gaussian_type', type=str, default='standard', help='type of Gaussians used to define generalized chi^2')
	parser.add_argument('--gaussian_scaling', type=float, default=1.0, help='scaling of Gaussians used to define generalized chi^2')
	parser.add_argument('--dimension', type=int, default=10, help='dimension of multivariate Gaussian used to define generalized chi^2')
	parser.add_argument('--quantile_upper_bound', type=float, default=10.0, help='largest quantile to compute probability / bound for')
	parser.add_argument('--num_quantiles', type=int, default=10, help='number of quantiles to evaluate on')
	parser.add_argument('--verbose', type=bool, default=True, help='verbose mode (True/False)')
	parser.add_argument('--combined', type=bool, default=True, help='whether to overlay Markov and chi^2 tail bounds')
	args = parser.parse_args()

	quantiles = np.linspace(1e-3, args.quantile_upper_bound, args.num_quantiles + 1)

	print(50 * "*")
	print("Evaluating bounds...")

	means, variances = get_generalized_chi_2_params(args.gaussian_type, args.dimension, scaling=args.gaussian_scaling)
	unreduced_gaussian_samples = sample_unreduced_gaussians(means, variances, args.num_samples)
	reduced_generalized_chi_2_samples = gaussians_to_generalized_chi_2s(unreduced_gaussian_samples)

	estimated_probabilities = []
	estimated_shifted_scaled_probabilities = []
	markov_bounds = []
	chi_2_bounds = []
	for quantile in quantiles:
		estimated_probabilities.append(get_estimated_probability(reduced_generalized_chi_2_samples, quantile))
		markov_bounds.append(compute_markov_bound(means, variances, quantile))
		estimated_shifted_probability, estimated_shifted_scaled_probability, shifted_scaled_quantile = get_estimated_shifted_scaled_probabilities(unreduced_gaussian_samples, means, variances, quantile)
		estimated_shifted_scaled_probabilities.append((estimated_shifted_probability, estimated_shifted_scaled_probability))
		chi_2_bounds.append(compute_chi_2_bound(args.dimension, shifted_scaled_quantile))
	
	if args.verbose:
		estimated_shifted_probabilities = [estimated_shifted_scaled_probabilities[index][0] for index in range(len(estimated_shifted_scaled_probabilities))]
		estimated_shifted_scaled_probabilities = [estimated_shifted_scaled_probabilities[index][1] for index in range(len(estimated_shifted_scaled_probabilities))]
		print("Quantiles: {}".format(np.round(quantiles, 2)))
		print("Estimated probabilities: {}".format(np.round(estimated_probabilities, 2)))
		print("Upper bound estimated probabilities: {}".format(np.round(estimated_shifted_probabilities, 2)))
		print("Upper upper bound estimated probabilities: {}".format(np.round(estimated_shifted_scaled_probabilities, 2)))
		print("Markov bounds: {}".format(np.round(markov_bounds, 2)))
		print("Chi_2 bounds: {}".format(np.round(chi_2_bounds, 2)))

	generate_markov_probability_bound_plot(quantiles, markov_bounds, reduced_generalized_chi_2_samples, estimated_probabilities, args.gaussian_type, scaling=args.gaussian_scaling)

	generate_chi_2_probability_bound_plot(quantiles, chi_2_bounds, markov_bounds, reduced_generalized_chi_2_samples, estimated_probabilities, args.gaussian_type, scaling=args.gaussian_scaling, df=args.dimension, combined=args.combined)

	print("Finished evaluating bounds...")
	print(50 * "*")

