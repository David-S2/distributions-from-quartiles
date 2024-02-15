import numpy as np
from scipy.stats import skewnorm
from scipy.stats import gamma
from scipy.stats import weibull_min
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import numbers

class Pdf_from_quartiles:
	def __init__(self, q1, q2, q3):
		self.qs = [q1, q2, q3]
		self.valid_entries = self.__quartiles_are_valid()
		self.skewnorm = Skewnorm_from_quartiles(self.qs)
		self.gamma = Gamma_from_quartiles(self.qs)
		self.weibull = Weibull_from_quartiles(self.qs)
		self.print_means()
		self.print_proportion_negative()
		self.plot()

	def __str__(self):
		string = "The following distributions have been successfully generated:"
		valid = False
		if self.skewnorm.valid:
			string = string + "\n - Skew normal distribution"
			valid = True
		if self.gamma.valid:
			string = string + "\n - Gamma distribution"
			valid = True
		if self.gamma.valid:
			string = string + "\n - Weibull distribution"
			valid = True
		if not valid:
			return "No distributions were successfully generated."
		else: return string
	
	def print_proportion_negative(self):
		string = "Proportions negative:"
		valid = False
		if self.skewnorm.valid:
			string = string + "\n - Skew normal distribution: " \
				+ str(round(self.skewnorm.proportion_negative * 100, 1)) + "%"
			valid = True
		if self.gamma.valid:
			string = string + "\n - Gamma distribution: " \
				+ str(round(self.gamma.proportion_negative * 100, 1)) + "%"
			valid = True
		if self.gamma.valid:
			string = string + "\n - Weibull distribution: " \
				+ str(round(self.weibull.proportion_negative * 100, 1)) + "%"
			valid = True
		if valid:
			print(string)

	def print_means(self):
		string = "Means:"
		valid = False
		if self.skewnorm.valid:
			string = string + "\n - Skew normal distribution: " \
				+ str(self.skewnorm.stats["mean"])
			valid = True
		if self.gamma.valid:
			string = string + "\n - Gamma distribution: " \
				+ str(self.gamma.stats["mean"])
			valid = True
		if self.gamma.valid:
			string = string + "\n - Weibull distribution: " \
				+ str(self.weibull.stats["mean"])
			valid = True
		if valid:
			print(string)

	def __quartiles_are_valid(self):
		valid = False
		if self.__quartiles_are_numeric():
			if self.__quartiles_are_valid_numbers():
				valid = True
		return valid
			
	def __quartiles_are_numeric(self):
		if \
			not isinstance(self.qs[0], numbers.Real) or \
			not isinstance(self.qs[1], numbers.Real) or \
			not isinstance(self.qs[2], numbers.Real):
			print("The quartiles must be real numbers")
			return False
		else:
			return True

	def __quartiles_are_valid_numbers(self):
		if self.qs[0] == self.qs[1]:
			print(
				"The quartiles provided are invalid. The median cannot have the same value as " \
					"the lower quartile."
			)
			return False
		elif self.qs[1] == self.qs[2]:
			print(
				"The quartiles provided are invalid. The median cannot have the same value as " \
					"the upper quartile."
			)
			return False
		elif self.qs[2] < self.qs[1] and self.qs[2] < self.qs[1]:
			print(
				"The quartiles provided are invalid. The upper and lower quartiles cannot " \
					"both be less than the median."
			)
			return False
		elif self.qs[0] > self.qs[1] and self.qs[2] > self.qs[1]:
			print(
				"The quartiles provided are invalid. The upper and lower quartiles cannot " \
					"both be greater than the median."
			)
			return False
		else:
			return True

	def __distribution_is_valid(self, dist):
		if not dist.valid:
			print(dist.message)
			return False
		else:
			return True

	def __create_plot(self, title, x_label, y_label, dist, plot_type):
		plt.axvline(x=0, color='black', linestyle='-')
		plt.title(title)
		plt.xlabel(x_label)
		plt.ylabel(y_label)
		plt.axvspan(
			xmin=dist.stats["mean"] - dist.stats["sd"],
			xmax=dist.stats["mean"] + dist.stats["sd"],
			color='blue',
			alpha=0.1,
			label="Standard deviation"
    )
		plt.axvspan(
			xmin=self.qs[0],
			xmax=self.qs[2],
			color='red',
			alpha=0.1,
			label="Interquartile range"
    )
		plt.axvline(x=self.qs[1], color='red', linestyle='--', label='Median')
		plt.axvline(x=dist.stats["mean"], color='blue', linestyle='--', label='Mean')
		if(plot_type == "pdf"):
			plt.plot(dist.x_values, dist.pdf_values)
		else:
			plt.plot(dist.x_values, dist.cdf_values)
		plt.legend()
		plt.show()
	
	def __plot_all(self):
		plt.axvline(x=0, color='black', linestyle='-')
		plt.title("Generated PDFs")
		plt.xlabel("X")
		plt.ylabel("Probability Density")
		plt.axvspan(
			xmin=self.qs[0],
			xmax=self.qs[2],
			color='red',
			alpha=0.1,
			label="Interquartile range"
    )
		plt.axvline(x=self.qs[1], color='red', linestyle='--', label='Median')
		if self.skewnorm.valid:
			plt.plot(self.skewnorm.x_values, self.skewnorm.pdf_values, label="Skew normal")
		if self.gamma.valid:
			plt.plot(self.gamma.x_values, self.gamma.pdf_values, label="Gamma")
		if self.weibull.valid:
			plt.plot(self.weibull.x_values, self.weibull.pdf_values, label="Weibull")
		plt.legend()
		plt.show()

	def plot(self, distribution_name="all"):
		match distribution_name:
			case "skewnorm":
				if self.__distribution_is_valid(self.skewnorm):
					self.__create_plot(
						"Skew Normal Distribution PDF",
						"X",
						"Probability Density",
						self.skewnorm,
						"pdf"
					)
					self.__create_plot(
						"Skew Normal Distribution CDF",
						"X",
						"Cumulative Probability Density",
						self.skewnorm,
						"cdf"
					)
			case "gamma":
				if self.__distribution_is_valid(self.gamma):
					self.__create_plot(
						"Gamma Distribution PDF",
						"X",
						"Probability Density",
						self.gamma,
						"pdf"
					)
					self.__create_plot(
						"Gamma Distribution CDF",
						"X",
						"Cumulative Probability Density",
						self.gamma,
						"cdf"
					)
			case "weibull":
				if self.__distribution_is_valid(self.weibull):
					self.__create_plot(
						"Weibull Distribution PDF",
						"X",
						"Probability Density",
						self.weibull,
						"pdf"
					)
					self.__create_plot(
						"Weibull Distribution CDF",
						"X",
						"Cumulative Probability Density",
						self.weibull,
						"cdf"
					)
			case _ :
				if self.skewnorm.valid or self.gamma.valid or self.weibull.valid:
					self.__plot_all()
				else:
					print("No distributions were successfully generated.")
	
	def stats(self, dist=""):
		match dist:
			case "skewnorm":
				if self.__distribution_is_valid(self.skewnorm):
					return self.skewnorm.stats
				else: return {}
			case "gamma":
				if self.__distribution_is_valid(self.gamma):
					return self.gamma.stats
				else: return {}
			case "weibull":
				if self.__distribution_is_valid(self.weibull):
					return self.weibull.stats
				else: return {}
			case _ :
				print(
					"No distribution identified, please pass 'skewnorm', 'gamma' or " \
					"'weibull' as an argument."
				)
				return {}
	
	def params(self, dist=""):
		match dist:
			case "skewnorm":
				if self.__distribution_is_valid(self.skewnorm):
					return self.skewnorm.params
				else: return {}
			case "gamma":
				if self.__distribution_is_valid(self.gamma):
					return self.gamma.params
				else: return {}
			case "weibull":
				if self.__distribution_is_valid(self.weibull):
					return self.weibull.params
				else: return {}
			case _ :
				print(
					"No distribution identified, please pass 'skewnorm', 'gamma' or " \
					"'weibull' as an argument."
				)
				return {}

class Skewnorm_from_quartiles:
	def __init__(self, qs):
		self.qs = qs
		self.__get_distribution()
	
	def __get_quartiles(self, alpha, loc=0, scale=1):
		return [
			skewnorm.ppf(0.25, alpha, loc, scale),
			skewnorm.ppf(0.5, alpha, loc, scale),
			skewnorm.ppf(0.75, alpha, loc, scale),
		]

	def __get_shape_parameter(self):
		target_ratio = (self.qs[1] - self.qs[0]) / (self.qs[2] - self.qs[1])
		def objective_function(alpha):
			quartiles = self.__get_quartiles(alpha)
			return (quartiles[1] - quartiles[0]) / (quartiles[2] - quartiles[1]) \
				- target_ratio
		if self.qs[2] - self.qs[1] > self.qs[1] - self.qs[0]:
			return root_scalar(objective_function, x0=1, x1=1.1).root
		else:
			return root_scalar(objective_function, x0=1, x1=0.9).root
	
	def __get_scale_parameter(self):
		target_iqr = self.qs[2] - self.qs[0]
		def objective_function(scale):
			quartiles = self.__get_quartiles(self.params["alpha"], 0,	scale)
			return quartiles[2] - quartiles[0] - target_iqr
		return root_scalar(
			objective_function,
			x0 = (self.qs[2] - self.qs[0]) / 4, x1 = (self.qs[2] - self.qs[0]) / 2
		).root

	def __get_stats(self):
		self.stats = {}
		self.stats["mean"], \
		self.stats["var"], \
		self.stats["skewness"], \
		self.stats["kurtosis"], \
		= skewnorm.stats(
			self.params["alpha"],
			self.params["loc"],
			self.params["scale"],
			moments="mvsk"
		)
		self.stats["sd"] = np.sqrt(self.stats["var"])

	def __quartiles_define_valid_skewnorm(self):
		if (self.qs[1]-self.qs[0]) / (self.qs[2]-self.qs[1]) < 0.748344:
			self.valid = False
			self.message = "This distribution is too skewed to define a skew normal " \
				+ "distribution. No skew normal distribution exists where the ratio " \
				+ "(Q2 - Q1) / (Q3 - Q2) is less than 0.748344. The ratio in this case is " \
				+ str(round((self.qs[1] - self.qs[0]) / (self.qs[2] - self.qs[1]), 6)) + "."
		elif (self.qs[1]-self.qs[0]) / (self.qs[2]-self.qs[1]) > 1.336284:
			self.valid = False
			self.message = "This distribution is too skewed to define a skew normal " \
				+ "distribution. No skew normal distribution exists where the ratio " \
				+ "(Q2 - Q1) / (Q3 - Q2) is greater than 1.336284. The ratio in this case " \
				+ "is " + str(round((self.qs[1] - self.qs[0]) / (self.qs[2] - self.qs[1]), 6)) + "."
		else:
			self.valid = True

	def __get_distribution(self):
		self.__quartiles_define_valid_skewnorm()
		if self.valid == True:
			self.params = {
				"alpha": self.__get_shape_parameter()
			}
			self.params["scale"] = self.__get_scale_parameter()
			self.params["loc"] = self.qs[1] \
				- skewnorm.ppf(
						0.5,
						self.params["alpha"],
						0,
						self.params["scale"]
					)
			self.__get_stats()
	
	@property
	def x_values(self):
		return np.linspace(
			self.qs[0] - 3 * (self.qs[1] - self.qs[0]),
			self.qs[2] + 3 * (self.qs[2] - self.qs[1]),
			1000
		)
	
	@property
	def pdf_values(self):
		return skewnorm.pdf(
				self.x_values,
				self.params["alpha"],
				loc=self.params["loc"],
				scale=self.params["scale"]
			)
	
	@property
	def cdf_values(self):
		return skewnorm.cdf(
				self.x_values,
				self.params["alpha"],
				loc=self.params["loc"],
				scale=self.params["scale"]
			)
	
	@property
	def proportion_negative(self):
		return skewnorm.cdf(
				0,
				self.params["alpha"],
				loc=self.params["loc"],
				scale=self.params["scale"]
			)

class Gamma_from_quartiles:
	def __init__(self, qs):
		self.qs = qs
		self.__get_distribution()
	
	def __get_quartiles(self, alpha, loc=0, scale=1):
		return [
			gamma.ppf(0.25, alpha, loc, scale),
			gamma.ppf(0.5, alpha, loc, scale),
			gamma.ppf(0.75, alpha, loc, scale),
		]

	def __get_shape_parameter(self):
		target_ratio = (self.qs[2] - self.qs[1]) / (self.qs[1] - self.qs[0])
		def objective_function(alpha):
			quartiles = self.__get_quartiles(alpha)
			return (quartiles[2] - quartiles[1]) / (quartiles[1] - quartiles[0]) \
				- target_ratio
		if target_ratio < 100:
			return root_scalar(objective_function, x0=0.087891, x1=0.2).root
		else:
			return root_scalar(objective_function, x0=0.058707, x1=0.1).root
	
	def __get_scale_parameter(self):
		target_iqr = self.qs[2] - self.qs[0]
		def objective_function(scale):
			quartiles = self.__get_quartiles(self.params["alpha"], 0,	scale)
			return quartiles[2] - quartiles[0] - target_iqr
		return root_scalar(objective_function, x0=10, x1=1000).root

	def __get_stats(self):
		self.stats = {}
		self.stats["mean"], \
		self.stats["var"], \
		self.stats["skewness"], \
		self.stats["kurtosis"], \
			= gamma.stats(
			self.params["alpha"],
			self.params["loc"],
			self.params["scale"],
			moments="mvsk"
		)
		self.stats["sd"] = np.sqrt(self.stats["var"])

	def __quartiles_define_valid_gamma(self):
		if self.qs[1]-self.qs[0] == self.qs[2]-self.qs[1]:
			self.valid = False
			self.message = "This distribution is symmetric, Gamma distributions cannot be " \
				+ "symmetric."
		elif self.qs[2]-self.qs[1] < self.qs[1]-self.qs[0]:
			self.valid = False
			self.message = "This distribution is negatively skewed, Gamma distributions can " \
				+ "only be generated for positively skewed data."
		elif (self.qs[2]-self.qs[1]) / (self.qs[1]-self.qs[0]) > 1000:
			self.valid = False
			self.message = "This distribution is too skewed, Gamma distributions cannot " \
				+ "be generated where the ratio (Q3 - Q2) / (Q2 - Q1) is greater than 1000. " \
				+ "The ratio in this case is " \
				+ str(round((self.qs[2] - self.qs[1]) / (self.qs[1] - self.qs[0]), 0)) + "."
		else:
			self.valid = True

	def __get_distribution(self):
		self.__quartiles_define_valid_gamma()
		if self.valid == True:
			self.params = {
				"alpha": self.__get_shape_parameter()
			}
			self.params["scale"] = self.__get_scale_parameter()
			self.params["loc"] = self.qs[1] \
				- gamma.ppf(
						0.5,
						self.params["alpha"],
						0,
						self.params["scale"]
					)
			self.__get_stats()
	
	@property
	def x_values(self):
		return np.linspace(
			self.params["loc"],
			self.qs[2] + 3 * (self.qs[2] - self.qs[1]),
			1000
		)
	
	@property
	def pdf_values(self):
		return gamma.pdf(
				self.x_values,
				self.params["alpha"],
				loc=self.params["loc"],
				scale=self.params["scale"]
			)
	
	@property
	def cdf_values(self):
		return gamma.cdf(
				self.x_values,
				self.params["alpha"],
				loc=self.params["loc"],
				scale=self.params["scale"]
			)

	@property
	def proportion_negative(self):
		return gamma.cdf(
				0,
				self.params["alpha"],
				loc=self.params["loc"],
				scale=self.params["scale"]
			)

class Weibull_from_quartiles:
	def __init__(self, qs):
		self.qs = qs
		self.__get_distribution()
	
	def __get_quartiles(self, alpha, loc=0, scale=1):
		return [
			weibull_min.ppf(0.25, alpha, loc, scale),
			weibull_min.ppf(0.5, alpha, loc, scale),
			weibull_min.ppf(0.75, alpha, loc, scale),
		]

	def __get_shape_parameter(self):
		target_ratio = (self.qs[2] - self.qs[1]) / (self.qs[1] - self.qs[0])
		def objective_function(alpha):
			quartiles = self.__get_quartiles(alpha)
			return (quartiles[2] - quartiles[1]) / (quartiles[1] - quartiles[0]) \
				- target_ratio
		return root_scalar(objective_function, x0=0.1003314, x1=0.2).root
	
	def __get_scale_parameter(self):
		target_iqr = self.qs[2] - self.qs[0]
		def objective_function(scale):
			quartiles = self.__get_quartiles(self.params["alpha"], 0,	scale)
			return quartiles[2] - quartiles[0] - target_iqr
		return root_scalar(objective_function, x0=0.1, x1=1000).root

	def __get_stats(self):
		self.stats = {}
		self.stats["mean"], \
		self.stats["var"], \
		self.stats["skewness"], \
		self.stats["kurtosis"], \
			= weibull_min.stats(
			self.params["alpha"],
			self.params["loc"],
			self.params["scale"],
			moments="mvsk"
		)
		self.stats["sd"] = np.sqrt(self.stats["var"])

	def __quartiles_define_valid_weibull(self):
		if (self.qs[2]-self.qs[1]) / (self.qs[1]-self.qs[0]) > 1000:
			self.valid = False
			self.message = "This distribution is too skewed, Weibull distributions cannot " \
				+ "be generated where the ratio (Q3 - Q2) / (Q2 - Q1) is greater than 1000. " \
				+ "The ratio in this case is " \
				+ str(round((self.qs[2] - self.qs[1]) / (self.qs[1] - self.qs[0]), 0)) + "."
		elif (self.qs[2]-self.qs[1]) / (self.qs[1]-self.qs[0]) < 0.788223:
			self.valid = False
			self.message = "This distribution is too skewed, no Weibull distributions exists" \
				+ " where the ratio (Q3 - Q2) / (Q2 - Q1) is less than 0.788223. The ratio " \
				+ "in this case is " \
				+ str(round((self.qs[2] - self.qs[1]) / (self.qs[1] - self.qs[0]), 1)) + "."
		else:
			self.valid = True

	def __get_distribution(self):
		self.__quartiles_define_valid_weibull()
		if self.valid == True:
			self.params = {
				"alpha": self.__get_shape_parameter()
			}
			self.params["scale"] = self.__get_scale_parameter()
			self.params["loc"] = self.qs[1] \
				- weibull_min.ppf(
						0.5,
						self.params["alpha"],
						0,
						self.params["scale"]
					)
			self.__get_stats()
	
	@property
	def x_values(self):
		return np.linspace(
			self.params["loc"],
			self.qs[2] + 3 * (self.qs[2] - self.qs[1]),
			1000
		)
	
	@property
	def pdf_values(self):
		return weibull_min.pdf(
				self.x_values,
				self.params["alpha"],
				loc=self.params["loc"],
				scale=self.params["scale"]
			)
	
	@property
	def cdf_values(self):
		return weibull_min.cdf(
				self.x_values,
				self.params["alpha"],
				loc=self.params["loc"],
				scale=self.params["scale"]
			)
	
	@property
	def proportion_negative(self):
		return weibull_min.cdf(
				0,
				self.params["alpha"],
				loc=self.params["loc"],
				scale=self.params["scale"]
			)