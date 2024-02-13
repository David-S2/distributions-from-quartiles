import numpy as np
from scipy.stats import gamma
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import numbers

def get_gamma_quartiles(a, loc=0, scale=1):
    return [
        gamma.ppf(0.25, a, loc, scale),
        gamma.ppf(0.5, a, loc, scale),
        gamma.ppf(0.75, a, loc, scale),
    ]

def get_gamma_from_quartiles(q1, q2, q3):
    shape_target = (q3 - q2) / (q2 - q1)
    def objective_function_1(alpha):
        quartiles = get_gamma_quartiles(alpha)
        return (quartiles[2] - quartiles[1]) / (quartiles[1] - quartiles[0]) \
            - shape_target
    if shape_target < 100:
        alpha = root_scalar(objective_function_1, x0=0.087891, x1=0.2).root
    elif shape_target < 1000:
        alpha = root_scalar(objective_function_1, x0=0.058707, x1=0.1).root
    else:
        print("It is not possible to generate a gamma distribution where (Q3-Q2)/(Q2-Q1) > 1000")
    iqr_target = q3 - q1
    def objective_function_2(scale):
        quartiles = get_gamma_quartiles(alpha, 0, scale)
        return quartiles[2] - quartiles[0] - iqr_target
    scale = root_scalar(objective_function_2, x0=10, x1=1000).root
    return {
        "alpha": alpha,
        "scale": scale,
        "loc": 0
    }

def plot_pdf(params, descriptive_stats, q1, q2, q3):
    x_values = np.linspace(0, q3 + 3 * (q3 - q1), 1000)
    pdf_values = gamma.pdf(x_values, params["alpha"], loc=params["loc"], scale=params["scale"])
    plt.title('Gamma Distribution PDF')
    plt.xlabel('X')
    plt.ylabel('Probability Density')
    plt.axvspan(
        xmin=descriptive_stats["mean"] - descriptive_stats["standard_deviation"],
        xmax=descriptive_stats["mean"] + descriptive_stats["standard_deviation"],
        color='blue',
        alpha=0.1,
        label="Standard deviation"
    )
    plt.axvspan(
        xmin=q1,
        xmax=q3,
        color='red',
        alpha=0.1,
        label="Interquartile range"
    )
    plt.axvline(x=q2, color='red', linestyle='--', label='Median')
    plt.axvline(x=descriptive_stats["mean"], color='blue', linestyle='--', label='Mean')
    plt.plot(x_values, pdf_values)
    plt.legend()
    plt.show()

def plot_cdf(params, descriptive_stats, q1, q2, q3):
    x_values = np.linspace(0, q3 + 3 * (q3 - q1), 1000)
    cdf_values = gamma.cdf(x_values, params["alpha"], loc=params["loc"], scale=params["scale"])
    plt.title('Cumulative Probability Density Function')
    plt.xlabel('X')
    plt.ylabel('Probability Density')
    plt.axvspan(
        xmin=descriptive_stats["mean"] - descriptive_stats["standard_deviation"],
        xmax=descriptive_stats["mean"] + descriptive_stats["standard_deviation"],
        color='blue',
        alpha=0.1,
        label="Standard deviation"
    )
    plt.axvspan(
        xmin=q1,
        xmax=q3,
        color='red',
        alpha=0.1,
        label="Interquartile range"
    )
    plt.axvline(x=q2, color='red', linestyle='--', label='Median')
    plt.axvline(x=descriptive_stats["mean"], color='blue', linestyle='--', label='Mean')
    plt.plot(x_values, cdf_values)
    plt.legend()
    plt.show()

def quartiles_are_non_numeric(q1, q2, q3):
    if \
        not isinstance(q1, numbers.Real) or \
        not isinstance(q2, numbers.Real) or \
        not isinstance(q3, numbers.Real):
        return {
            "fail": True,
            "message": "The quartiles provided must be real numbers."
        }
    else: return { "fail": False }

def quartiles_are_invalid(q1, q2, q3):
    if q1 == q2:
        return {
            "fail": True,
            "message": "The quartiles provided are invalid. The median cannot have " \
                "the same value as the lower quartile."
        }
    elif q2 == q3:
        return {
            "fail": True,
            "message": "The quartiles provided are invalid. The median cannot have " \
                "the same value as the upper quartile."
        }
    elif q1 < q2:
        if q3 < q2:
            return {
                "fail": True,
                "message": "The quartiles provided are invalid. The upper and lower " \
                    "quartiles cannot both be less than the median."
            }
        else: return { "fail": False }
    elif q1 > q2:
        if q3 > q2:
            return {
                "fail": True,
                "message": "The quartiles provided are invalid. The upper and lower " \
                    "quartiles cannot both be greater than the median."
            }
        else: return { "fail": False }
    return { "fail": False }

def distribution_could_be_lognormal(q1, q2, q3):
    if q3 - q2 > q2 - q1:
        thresh = 0.2
        logQ1 = np.log(q1)
        logQ2 = np.log(q2)
        logQ3 = np.log(q3)
        logQ2_to_logQ3 = logQ3 - logQ2
        logQ1_to_logQ2 = logQ2 - logQ1
        if logQ2_to_logQ3 > logQ1_to_logQ2:
            diff = (logQ2_to_logQ3 - logQ1_to_logQ2) / logQ1_to_logQ2
            if diff < thresh:
                print(
                    "This distribution could be lognormal, the difference between " \
                    + "log(Q3) and log(Q2) is only " + str(round(diff * 100, 1)) + "% " \
                    + "greater than the difference between log(Q2) and log(Q1)."
                )
        if logQ1_to_logQ2 > logQ2_to_logQ3:
            diff = (logQ1_to_logQ2 - logQ2_to_logQ3) / logQ2_to_logQ3
            if diff < thresh:
                print(
                    "This distribution could be lognormal, the difference between " \
                    + "log(Q2) and log(Q3) is only " + str(round(diff * 100, 1)) + "% " \
                    + "greater than the difference between log(Q3) and log(Q2)."
                )

def get_gamma_dist_stats_from_quartiles(q1, q2, q3):
    test = quartiles_are_non_numeric(q1, q2, q3)
    if test["fail"]: return test
    test = quartiles_are_invalid(q1, q2, q3)
    if test["fail"]: return test
    distribution_could_be_lognormal(q1, q2, q3)
    if test["fail"]: return test
    params = get_gamma_from_quartiles(q1, q2, q3)
    descriptive_stats = {}
    descriptive_stats["mean"], \
    descriptive_stats["var"], \
    descriptive_stats["skewness"], \
    descriptive_stats["kurtosis"] = \
        gamma.stats(params["alpha"], params["loc"], params["scale"], moments="mvsk")
    descriptive_stats["standard_deviation"] = np.sqrt(descriptive_stats["var"])
    plot_pdf(params, descriptive_stats, q1, q2, q3)
    plot_cdf(params, descriptive_stats, q1, q2, q3)
    print(get_gamma_quartiles(params["alpha"], params["loc"], params["scale"]))
    return {
        "gamma_dist": params,
        "descriptive_stats": descriptive_stats,
    }