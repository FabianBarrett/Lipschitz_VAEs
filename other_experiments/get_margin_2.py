import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def compute_C_m(m):
	return (1 / np.sqrt(np.pi)) * np.exp(0.5 * (m - (m - 1) * np.log(m)))

def compute_u(r, a, b, gamma, x):
	numerator = ((r / a) - b * x) ** 2
	denominator = 2 * (gamma ** 2)
	return numerator / denominator

def compute_main_term(u, m):
	numerator = np.power(u, 0.5 * m) * np.exp(-0.5 * u)
	denominator = u - m + 2
	return numerator / denominator

# Note: Expression minus 0.5
def get_margin_expression(args, x):
	# Compute quantile
	u = compute_u(args['r'], args['a'], args['b'], args['gamma'], x)
	if u <= (args['m'] - 2):
		return np.inf
	# Get the coefficient that depends on m in the bound
	C_m = compute_C_m(args['m'])
	# Main term
	main_term = compute_main_term(u, args['m'])
	# Want to make sure this expression is less than or equal to 0
	full_expression = C_m * main_term - 0.5
	return full_expression

def compute_bound(u, m):
	C_m = compute_C_m(m)
	main_term = compute_main_term(u, m)
	bound = 1 - C_m * main_term
	return bound

def get_margin(args):
	encoder_standard_deviation_norm = np.sqrt(np.sum(np.array([args['gamma'] ** 2 for _ in range(args['m'])])))
	print("Encoder standard deviation norm for gamma={}: {}".format(args['gamma'], encoder_standard_deviation_norm))
	first_perturbation_upper_bound = args['r'] / (args['a'] * args['b'])
	x_values = np.linspace(0, first_perturbation_upper_bound, num=100)
	y_values = np.array([get_margin_expression(args, x_value) for x_value in x_values])
	solution = (-1)
	for index in range(len(y_values)):
		if y_values[index] <= 0 and x_values[index] > solution:
			solution = x_values[index]
	if solution == (-1):
		print("Solution: {}".format(0.0))
	else:
		print("Solution: {}".format(solution))
	plt.plot(x_values[y_values < np.inf], y_values[y_values < np.inf])
	plt.show()
	print("First upper bound: {}".format(first_perturbation_upper_bound))
	print("Valid x-values: {}".format(x_values[y_values < np.inf]))
	print("Valid y-values: {}".format(y_values[y_values < np.inf]))

def plot_bound(args):
	colors = [color for color in mcolors.TABLEAU_COLORS][:5]
	possible_u_values = np.linspace(0, 20, num=100)
	m_values = [m_value for m_value in range(2, 12, 2)]
	for m_index in range(len(m_values)):
		u_values = []
		bound_values = []
		for u_index in range(len(possible_u_values)):
			if possible_u_values[u_index] > (m_values[m_index] - 2):
				u_values.append(possible_u_values[u_index])
				bound_values.append(compute_bound(possible_u_values[u_index], m_values[m_index]))
		plt.plot(u_values, bound_values, color=colors[m_index], label=r'$m$={}'.format(m_values[m_index]))
	plt.title(r"Probability Bound 2 as a function of $u$")
	plt.legend()
	plt.ylim(0, 1)
	plt.xlabel(r"$u$")
	plt.ylabel("Probability Bound 2")
	plt.savefig(os.getcwd() + "/out/vae/other_figures/probability_bound_2_u_plot.png", dpi=300)

def main(args):
	plot_bound(args)
	get_margin(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='experiment with margin 2')
    parser.add_argument('--m', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--r', type=int)
    parser.add_argument('--a', type=int)
    parser.add_argument('--b', type=int)
    args = vars(parser.parse_args())

    print(args)

    main(args)


