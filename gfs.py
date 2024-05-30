"""
Helper function to compute, modify, analyse GFS.
"""

import numpy as np
from collections import Counter
from typing import List


def expected_gf(n: int, k: int, theta: int, rho: float) -> float:
    """Calculate the expected gene frequency for the class k.

    Parameters:
    -----------
        n: int :
            Population size.
        k: int :
            Gene frequency class.
        theta: int :
            Gene-gain rate-parameter.
        rho: float :
            Gene-loss rate-parameter.

    Returns:
    --------
        egf: float :
            Expected gene frequency.
    """

    def log_factorial_ratio(n, k, rho):
        # Computes log(n * (n-1) * ... * (n-k+1) / ((n-1+rho) * (n-2+rho) * ... * (n-k+rho)))
        num_log_sum = sum(np.log(n - i) for i in range(k))
        den_log_sum = sum(np.log(n - 1 + rho - i) for i in range(k))
        return num_log_sum - den_log_sum

    theta = 2 * theta
    rho = 2 * rho

    log_factorial_term = log_factorial_ratio(n, k, rho)
    egf = (theta / k) * np.exp(log_factorial_term)
    return egf


def expected_gfs(n: int, theta: int, rho: float) -> List[float]:
    """Calculate the expected gene frequency spectrum.

    Parameters:
    -----------
        n: int :
            Population size.
        k: int :
            Gene frequency class.
        theta: int :
            Gene-gain rate-parameter.
        rho: float :
            Gene-loss rate-parameter.

    Returns:
    --------
        gfs: List[float] :
            Expected gene frequency spectrum.
    """

    return [expected_gf(n=n, k=k, theta=theta, rho=rho) for k in range(1, n + 1)]


def gfs_from_matrix(genotype: np.ndarray, num_samples: int) -> List[int]:
    """Computes the GFS from a gene matrix.

    Parameters:
    -----------
        genotype: np.ndarray :
            Gene matrix, Each row is a gene, each column an individual.
            0 for wildtype, 1 for gene gain.
        num_samples: int :
            Maximum number of time a gene may be present.

    Returns:
    --------
        gfs: List[int] :
            GFS from 1 to n.

    """
    genotype = np.array(genotype)
    genotype_sum = genotype.sum(axis=1)

    freq = Counter(genotype_sum)
    gfs = [freq.get(i, 0) for i in range(1, num_samples + 1)]

    return gfs


def padded_gfs_from_gm(lgm: List[List[int]], num_samples: int) -> np.ndarray:
    """
    Parameters:
    -----------
        lgm: List[List[int]] :
            Iterable of genotype matrices.
        num_samples: int :
            Number of samples.

    Returns:
    --------
        padded_gfs: np.ndarray[List[int]] :
            Array of padded Gene Frequency Spectra.
    """

    all_gfs = [gfs_from_matrix(gm, num_samples) for gm in lgm]
    pad = len(max(all_gfs, key=len))

    padded_gfs = np.array([np.pad(i, (0, pad - len(i))) for i in all_gfs])
    return padded_gfs


def normalise_gfs(gfs: List[float]) -> List[float]:
    """
    Normalises the GFS.

    Parameters:
    -----------
        gfs: List[float] :
            Gene Frequency Spectrum.
    Returns:
    --------
        gfs: List[float] :
            Normalised Gene frequency Spectrum.
    """
    total_genes = sum(gfs)
    return [g / total_genes for g in gfs]
