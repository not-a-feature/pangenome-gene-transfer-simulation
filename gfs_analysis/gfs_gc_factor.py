"""
Tests the impact of GC on the GFS of fixed trees.
"""

import gene_model
import gfs
import neutrality_test

import os
import numpy as np
import json

from multiprocessing import Pool

SELF_FN = os.path.dirname(os.path.abspath(__file__))
OUT_FN = os.path.join(SELF_FN, "data", "gfs", "gfs_gc_factor.csv")
OUT_PARAM_FN = os.path.join(SELF_FN, "data", "gfs", "gfs_gc_factor.json")
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
    nwk="",
):
    processes = 12

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
    theta = 1000
    rho = 0.2
    recomb = 0
    hgt = 0

    gene_conv_list = np.arange(0.0101, 0.0201, 0.0001)

    num_simulations = 12 * 20
    # Somewhat balanced
    # nwk = ((((A:0.05, B:0.05):0.01, (C:0.05, D:0.05):0.01):0.01,((E:0.05, F:0.05):0.01, (G:0.05, H:0.05):0.01):0.01):0.01, (I:0.01, J:0.01):0.07)

    # Unbalanced
    # nwk = "(((((((((A:0.05, B:0.05):0.001, C:0.051):0.001, D:0.052):0.001, E:0.053):0.037, F:0.09):0.01, G:0.10):0.01, H:0.11):0.01, I:0.12):0.01, J:0.13)"

    # Peak at Class 2 and 4 expected
    nwk = "((((A:0.01, B:0.01):0.01, (C:0.01, D:0.01):0.01):0.3,((E:0.01, F:0.01):0.01, (G:0.01, H:0.01):0.01):0.3):0.02, (I:0.01, J:0.01):0.33)"

    egfs = gfs.expected_gfs(n=num_samples, theta=theta, rho=rho)
    print(egfs)
    param = {
        "theta": theta,
        "rho": rho,
        "recomb": recomb,
        "hgt": hgt,
        "num_sites": num_sites,
        "num_samples": num_samples,
        "num_simulations": num_simulations,
        "gene_conv_list": list(gene_conv_list),
    }
    with open(OUT_PARAM_FN, "w") as f:
        json.dump(param, f)

    with open(OUT_FN, "w") as f:
        f.write("GC,Mean_Loss-Fixed,Mean_GFS_Fixed,Mean_Loss_Random,Mean_GFS_Random\n")

    for gene_conv in gene_conv_list:
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
            nwk="",
        )

        non_zero_mean_random_gfs = [g + 0.0000000001 for g in mean_gfs_random]
        mean_random_loss = neutrality_test.chi_squared_like_statistic(
            mean_gfs_fixed, non_zero_mean_random_gfs
        )

        out_str = f"{gene_conv:.6f},{mean_fixed_loss:.6f},{mean_gfs_fixed},{mean_random_loss:.6f},{mean_gfs_random}"
        out_str = out_str.replace("\n", " ")
        while "  " in out_str:
            out_str = out_str.replace("  ", " ")

        with open(OUT_FN, "a") as f:
            f.write(f"{out_str}\n")

        print(out_str)


if __name__ == "__main__":
    main()
