import numpy as np
from scipy.stats import skewnorm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import numbers

def get_skewnorm_quartiles(alpha, loc=0, scale=1):
    return [
        skewnorm.ppf(0.25, alpha, loc, scale),
        skewnorm.ppf(0.5, alpha, loc, scale),
        skewnorm.ppf(0.75, alpha, loc, scale),
    ]

def get_skewnorm_from_quartiles(q1, q2, q3):
    def objective_function_1(alpha):
        quartiles = get_skewnorm_quartiles(alpha)
        return (quartiles[1] - quartiles[0]) \
            / (quartiles[2] - quartiles[1]) \
            - (q2 - q1) / (q3 - q2)
    if q3 - q2 > q2 - q1:
        alpha = root_scalar(objective_function_1, x0=1, x1=1.1).root
    else:
        alpha = root_scalar(objective_function_1, x0=1, x1=0.9).root
    def objective_function_2(scale):
        quartiles = get_skewnorm_quartiles(alpha, 0, scale)
        return quartiles[2] - quartiles[0] - q3 + q1
    scale = root_scalar(objective_function_2, x0=(q3-q1)/4, x1=(q3-q1)/2).root
    median = get_skewnorm_median(alpha, 0, scale)
    return {
        "alpha": alpha,
        "scale": scale,
        "loc": q2 - get_skewnorm_median(alpha, 0, scale)
    }

def plot_pdf(params, descriptive_stats, q1, q2, q3):
    x_values = np.linspace(q1 - 3 * (q2 - q1), q3 + 3 * (q3 - q2), 1000)
    pdf_values = skewnorm.pdf(x_values, params["alpha"], loc=params["loc"], scale=params["scale"])
    plt.axvline(x=0, color='black', linestyle='-')
    plt.title('Skew Normal Distribution PDF')
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
    x_values = np.linspace(q1 - 3 * (q2 - q1), q3 + 3 * (q3 - q2), 1000)
    cdf_values = skewnorm.cdf(x_values, params["alpha"], loc=params["loc"], scale=params["scale"])
    plt.axvline(x=0, color='black', linestyle='-')
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

def distribution_is_too_skewed(q1, q2, q3):
    if (q2-q1) / (q3-q2) < 0.748344:
        return {
            "fail": True,
            "message": "This distribution is too skewed. It is not possible to " \
                + "generate skew normal distributions where the ratio " \
                + "(Q2 - Q1) / (Q3 - Q2) is less than 0.748344. The ratio in " \
                + "this case is " + str(round((q2-q1) / (q3-q2), 6)) + "."
        }
    elif (q2-q1) / (q3-q2) > 1.336284:
        return {
            "fail": True,
            "message": "This distribution is too skewed. It is not possible to " \
                + "generate skew normal distributions where the ratio " \
                + "(Q2 - Q1) / (Q3 - Q2) is greater than 1.336284. The ratio in " \
                + "this case is " + str(round((q2-q1) / (q3-q2), 6)) + "."
        }
    else:
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
                    + "log(Q2) and log(Q1) is only " + str(round(diff * 100, 1)) + "% " \
                    + "greater than the difference between log(Q3) and log(Q2)."
                )
    
def get_skewnorm_stats_from_quartiles(q1, q2, q3):
    test = quartiles_are_non_numeric(q1, q2, q3)
    if test["fail"]: return test
    test = quartiles_are_invalid(q1, q2, q3)
    if test["fail"]: return test
    test = distribution_is_too_skewed(q1, q2, q3)
    if test["fail"]: return test
    params = get_skewnorm_from_quartiles(q1, q2, q3)
    descriptive_stats = {}
    descriptive_stats["mean"], \
    descriptive_stats["var"], \
    descriptive_stats["skewness"], \
    descriptive_stats["kurtosis"] = \
        skewnorm.stats(params["alpha"], params["loc"], params["scale"], moments="mvsk")
    descriptive_stats["standard_deviation"] = np.sqrt(descriptive_stats["var"])
    plot_pdf(params, descriptive_stats, q1, q2, q3)
    plot_cdf(params, descriptive_stats, q1, q2, q3)
    return {
        "skew_normal": params,
        "descriptive_stats": descriptive_stats,
    }