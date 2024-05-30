from typing import List, Tuple, Union
import numpy as np
import scipy.stats as stats

import gene_model
import gfs

"""
This file contains the Neutrality Test Functions.
"""


def compute_kde_p_value(observed: float, assumed_dist: np.ndarray, alternative="greater") -> float:
    """Computes the p-value using Kernel Density Estimation (KDE) for a given observed value against an assumed distribution.

    Parameters
    ----------
    observed: float :
        The observed value.

    assumed_dist: np.ndarray :
        The assumed distribution as a numpy array.

    alternative: str :
        Defines the null and alternative hypotheses. Default is greater.


    Returns
    -------
    value: float:
        The p-value calculated.
    """

    kde = stats.gaussian_kde(assumed_dist)

    if alternative not in ["two-sided", "greater", "less"]:
        raise ValueError(f"Unexpected alternative {alternative}")

    p_left = kde.integrate_box_1d(-np.inf, observed)

    if alternative == "less":
        return p_left
    elif alternative == "greater":
        return 1 - p_left
    else:
        return min(p_left, 1 - p_left) * 2


def direct_min_p_value_kde(observed_gfs: List[float], all_gfs: List[np.ndarray]) -> float:
    """Computes the Bonferroni adjusted minimum p-value using KDE integration across multiple gene frequency distributions.

    Parameters
    ----------
    observed_gfs: List[float] :
        List of observed gene frequencies.

    all_gfs: List[np.ndarray] :
        List of expected gene frequency distributions.

    Returns
    -------
    value: float :
        The minimum p-value found.

    """
    gfs_distributions = all_gfs.transpose()
    probabilities = [
        compute_kde_p_value(observed, expected, alternative="two-sided")
        for observed, expected in zip(observed_gfs, gfs_distributions)
    ]
    return min(min(probabilities) * len(probabilities), 1)


def direct_min_p_value_ks(observed_gfs: List[float], all_gfs: List[np.ndarray]) -> float:
    """Computes the Bonferroni adjusted minimum p-value using Kolmogorov-Smirnov (KS) tests across multiple gene frequency distributions.


    Parameters
    ----------
    observed_gfs: List[float] :
        List of observed gene frequencies.

    all_gfs: List[np.ndarray] :
        List of expected gene frequency distributions.

    Returns
    -------
    value: float :
        The minimum p-value found.

    """
    gfs_distributions = all_gfs.transpose()
    probabilities = []
    for observed, expected in zip(observed_gfs, gfs_distributions):
        observed = np.array([observed])

        # Perform the KS test
        _, p_value = stats.kstest(observed, expected, alternative="two-sided")
        probabilities.append(p_value)

    return min(min(probabilities) * len(probabilities), 1)


def chi_squared_like_statistic(
    observed_gfs: List[float],
    expected_gfs: List[float],
    weights: Union[List[float]] = None,
) -> float:
    """
    Computes a chi-squared-like statistic for gene frequencies.

    Parameters
    ----------
    observed_gfs: List[float] :
        List of observed gene frequencies.

    expected_gfs: List[float] :
        Array of expected gene frequencies.

    weights: List[float] :
        Relative weights for each Gene Frequency Class.
        Default: [1, 1, ... , 1]

    Returns
    -------
    value: float :
        The chi-squared-like statistic.

    """
    observed_gfs = np.array(observed_gfs)
    expected_gfs = np.array(expected_gfs)

    if not all(expected_gfs):
        raise ZeroDivisionError("Expected GFS classes can't be zero.")

    if not len(observed_gfs) == len(expected_gfs):
        raise ValueError("Shape of observed and expected and gfs do not match.")

    if weights is None:
        weights = [1 for _ in range(len(expected_gfs))]

    if not round(np.sum(weights), 4) == len(expected_gfs):
        raise ValueError("Weights of chi-squared-like statistic must sum to one.")

    squared_error = np.square(observed_gfs - expected_gfs) / expected_gfs
    return np.sum(squared_error * weights)


def chi_square_like_p_value_kde(
    observed_gfs: List[int],
    all_gfs: List[np.ndarray],
    expected_gfs: np.ndarray,
) -> float:
    """
    Computes the probability using KDE integration and a chi-square-like approach for gene frequencies.

    Parameters
    ----------
    observed_gfs: List[int] :
        List of observed gene frequencies.

    all_gfs: List[np.ndarray] :
        List of all gene frequency simulations.

    expected_gfs: np.ndarray :
        Array of expected gene frequencies.

    Returns
    -------
    value: float :
        The p-value for the chi-square-like test.
    """
    chi_squared_values = [chi_squared_like_statistic(gf, expected_gfs) for gf in all_gfs]
    chi_squared_observed = chi_squared_like_statistic(observed_gfs, expected_gfs)
    return compute_kde_p_value(chi_squared_observed, chi_squared_values, alternative="greater")


def chi_square_like_p_value_ks(
    observed_gfs: List[int],
    all_gfs: List[np.ndarray],
    expected_gfs: np.ndarray,
) -> float:
    """
     Computes the probability using Kolmogorov-Smirnov (KS) tests and a chi-square-like approach for gene frequencies.

    Parameters
    ----------
    observed_gfs: List[int] :
        List of observed gene frequencies.

    all_gfs: List[np.ndarray] :
        List of simulated (expected) gene frequency spectra.

    expected_gfs: np.ndarray :
        Array of expected gene frequencies.

    Returns
    -------
    value: float :
        The p-value for the chi-square-like test.
    """

    chi_squared_observed = chi_squared_like_statistic(observed_gfs, expected_gfs)
    chi_squared_values = [chi_squared_like_statistic(gf, expected_gfs) for gf in all_gfs]

    chi_squared_observed = np.array([chi_squared_observed])
    # Perform the KS test
    _, p_value = stats.kstest(chi_squared_observed, chi_squared_values, alternative="less")

    return p_value


def test_neutrality(
    observed_gfs: List[int],
    num_samples: int,
    theta: int,
    rho: float,
    num_sites: int,
    num_simulations: int = 100,
    processes: int = 5,
) -> Tuple[float, float, float, float]:
    """Tests the neutrality of gene frequencies.

    Parameters
    ----------
    observed_gfs: List[int] :
        List of observed gene frequencies.

    num_samples: int :
        Size of the population.

    theta: int :
        Gene gain per time-step.

    rho: float :
        Gene loss frequency per gene.

    num_sites: int :
        Number of sites in the genome.

    num_simulations: int :
         (Default value = 100)
         Number of simulations. Increasing enhances the robustness.

    processes: int :
         (Default value = 5)
          Maximum number of parallel processes.

    Returns
    -------
    values: Tuple[float, float, float, float]:
        A tuple containing the probabilities:
         - Chi-square-like via KS test
         - Chi-square-like via KDE integration
         - Minimal direct via KS test
         - Minimal direct via KDE integration

    """
    lgm = gene_model.multi_genotype_matrices(
        n=num_simulations,
        processes=processes,
        num_samples=num_samples,
        theta=theta,
        rho=rho,
        num_sites=num_sites,
        double_site_relocation=True,
    )
    all_gfs = gfs.padded_gfs_from_gm(lgm, num_samples)

    expected_gfs = gfs.expected_gfs(n=num_samples, theta=theta, rho=rho)

    prob_chi_square_ks = chi_square_like_p_value_ks(observed_gfs, all_gfs, expected_gfs)
    prob_chi_square_kde = chi_square_like_p_value_kde(observed_gfs, all_gfs, expected_gfs)

    prob_direct_ks = direct_min_p_value_ks(observed_gfs, all_gfs)
    prob_direct_kde = direct_min_p_value_kde(observed_gfs, all_gfs)

    return prob_chi_square_ks, prob_chi_square_kde, prob_direct_ks, prob_direct_kde


if __name__ == "__main__":
    # Example Usage
    observed_gfs = [0, 53, 68, 18, 6, 21, 97, 45, 137, 622]
    num_samples = 10
    num_sites = 100000
    theta = 100
    rho = 0.1

    prob_chi_square_ks, prob_chi_square_kde, prob_direct_ks, prob_direct_kde = test_neutrality(
        observed_gfs,
        num_samples,
        theta,
        rho,
        num_sites,
        processes=14,
    )
    print("Chi-Square KS  :", prob_chi_square_ks)
    print("Chi-Square KDE :", prob_chi_square_kde)
    print("Direct KS:      ", prob_direct_ks)
    print("Direct KDE:     ", prob_direct_kde)
