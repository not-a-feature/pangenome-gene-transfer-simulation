"""
Tests the impact of HGT on the GFS of fixed trees.
"""

import gene_model
import gfs
import neutrality_test

import os
import numpy as np
import json

from multiprocessing import Pool

SELF_FN = os.path.dirname(os.path.abspath(__file__))
OUT_FN = os.path.join(SELF_FN, "data", "gfs", "gfs_hgt_factor.csv")
OUT_PARAM_FN = os.path.join(SELF_FN, "data", "gfs", "gfs_hgt_factor.json")

num_sites = 100000
alleles = ("absent", "present")


def compute_mts(
    num_simulations,
    num_samples,
    theta,
    rho,
    recomb,
    gene_conv,
    hgt_rate,
    nwk=None,
):
    processes = 50

    args = [
        [theta, rho, gene_conv, recomb, hgt_rate],
        nwk,
        num_samples,
    ]
    pool_args = [args for _ in range(num_simulations)]
    with Pool(processes=processes) as pool:
        result = pool.imap_unordered(simulate_gfs, pool_args)
        sim_gfs = list(result)

    sim_gfs = np.array(sim_gfs).mean(axis=0)
    return sim_gfs


def simulate_gfs(args):
    params, nwk, num_samples = args
    theta, rho, gene_conv, recomb, hgt_rate = params
    res = False
    while not res:
        try:
            mts = gene_model.gene_model(
                theta=theta,
                rho=rho,
                num_sites=num_sites,
                num_samples=num_samples,
                gene_conversion_rate=gene_conv,
                recombination_rate=recomb,
                hgt_rate=hgt_rate,
                ce_from_nwk=nwk,
                check_double_gene_gain=False,
            )
            res = True
        except Exception as e:
            print(e)
    gm = mts.genotype_matrix(alleles=alleles)
    sim_gfs = gfs.gfs_from_matrix(gm, num_samples)
    return sim_gfs


def main():
    num_samples = 10
    theta = 2000
    rho = 0.2
    gene_conv = 0
    recomb = 0

    hgt_list = np.arange(0.0, 0.001, 0.00001)

    num_simulations = 50 * 2
    # Peak at Class 2 and 4 expected
    nwk = "((((A:0.01, B:0.01):0.01, (C:0.01, D:0.01):0.01):0.3,((E:0.01, F:0.01):0.01, (G:0.01, H:0.01):0.01):0.3):0.02, (I:0.01, J:0.01):0.33)"

    egfs = gfs.expected_gfs(n=num_samples, theta=theta, rho=rho)
    print(egfs)
    param = {
        "theta": theta,
        "rho": rho,
        "recomb": recomb,
        "gene_conv": gene_conv,
        "num_sites": num_sites,
        "num_samples": num_samples,
        "num_simulations": num_simulations,
        "hgt_list": list(hgt_list),
    }
    with open(OUT_PARAM_FN, "w") as f:
        json.dump(param, f)

    with open(OUT_FN, "w") as f:
        f.write("HGT,Mean_Loss-Fixed,Mean_GFS_Fixed,Mean_Loss_Random,Mean_GFS_Random\n")

    for hgt in hgt_list:
        mean_gfs_fixed = compute_mts(
            num_simulations=num_simulations,
            num_samples=num_samples,
            theta=theta,
            rho=rho,
            recomb=recomb,
            gene_conv=gene_conv,
            hgt_rate=hgt,
            nwk=nwk,
        )
        mean_fixed_loss = neutrality_test.chi_squared_like_statistic(mean_gfs_fixed, egfs)

        mean_gfs_random = compute_mts(
            num_simulations=num_simulations,
            num_samples=num_samples,
            theta=theta,
            rho=rho,
            recomb=recomb,
            gene_conv=gene_conv,
            hgt_rate=hgt,
            nwk=None,
        )

        non_zero_mean_random_gfs = [g + 0.0000000001 for g in mean_gfs_random]
        mean_random_to_fixed_loss = neutrality_test.chi_squared_like_statistic(
            mean_gfs_fixed, non_zero_mean_random_gfs
        )

        mean_random_loss = neutrality_test.chi_squared_like_statistic(mean_gfs_random, egfs)

        out_str = f"{hgt:.6f},{mean_fixed_loss:.6f},{mean_gfs_fixed},{mean_random_to_fixed_loss:.6f},{mean_gfs_random},{mean_random_loss:.6f}"
        out_str = out_str.replace("\n", " ")
        while "  " in out_str:
            out_str = out_str.replace("  ", " ")

        with open(OUT_FN, "a") as f:
            f.write(f"{out_str}\n")

        out_str = f"{hgt:.6f},{mean_fixed_loss:.1f},{mean_random_to_fixed_loss:.1f},{mean_random_loss:.1f}"
        print(out_str)


if __name__ == "__main__":
    main()
