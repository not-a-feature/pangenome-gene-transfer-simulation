import os
import sys

SELF_FN = os.path.dirname(os.path.abspath(__file__))

parent_dir = os.path.abspath(os.path.join(SELF_FN, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from gene_mut import gene_model as gene_model
from gene_mut import gfs as gfs
from gene_mut import neutrality_test as neutrality_test

import numpy as np
from collections import Counter
import json

from matplotlib import pyplot as plt
from scipy import optimize
from multiprocessing import Pool
import time

import color_scheme

num_sites = 10000
alleles = ("absent", "present")
prefix = ""
OUT_PATH = os.path.join(SELF_FN, "data", "optimisation")

loss_list = []
processes = 12
num_simulations = 12

min_loss = float("inf")


def optim(average_num_genes, num_samples, ref_gfs, nwk):
    global min_loss
    min_loss = float("inf")
    bounds = [
        (10, 2000),  # theta
        (0.0001, 4),  # rho
        (0, 0.01),  # gene_conv
        (0, 0.01),  # recomb
        (0, 0.0005),  # hgt_rate
    ]

    minimum = optimize.differential_evolution(
        loss,
        args=(nwk, num_samples, ref_gfs),
        strategy="best1bin",
        bounds=bounds,
        maxiter=2000,
    )
    print()
    print(minimum)
    return minimum.x


def loss(x, nwk, num_samples, ref_gfs, return_gfs=False):
    global min_loss
    args = (x, nwk, num_samples, np.array(ref_gfs))

    print(
        f"Global Min Loss: {min_loss:.1f}, Computing: ",
        str(x).replace("\n", ""),
        end="",
    )

    pool_args = [args for _ in range(num_simulations)]
    with Pool(processes=processes) as pool:
        result = pool.imap_unordered(simulate_gfs, pool_args)
        sim_gfs = list(result)

    sim_gfs = np.array(sim_gfs).sum(axis=0)
    sim_gfs = sim_gfs / num_simulations

    # Weights, s.t. the edge classes are more important.
    # num_extra_weights = 10
    # single_weight = num_samples / (num_samples + num_extra_weights)
    # weights = [single_weight for _ in range(num_samples)]
    # weights[0] += 2 * single_weight
    # weights[1] += 2 * single_weight
    # weights[2] += single_weight
    # weights[-3] += single_weight
    # weights[-2] += 2 * single_weight
    # weights[-1] += 2 * single_weight

    # if return_gfs:
    #    weights = None
    weights = None
    ref_gfs = [r + 0.00001 for r in ref_gfs]
    local_loss = neutrality_test.chi_squared_like_statistic(sim_gfs, ref_gfs, weights=weights)
    min_loss = min(local_loss, min_loss)
    if min_loss == local_loss:
        with open(out_fn, "w") as f:
            d = {
                "theta": x[0],
                "rho": x[1],
                "gene_conversion": x[2],
                "recombination": x[3],
                "hgt_rate": x[4],
            }
            d = json.dumps(d, indent=3)
            f.write(d)

    print("\r", end="")
    print(
        f"Global Min Loss: {min_loss:.1f}, Local Loss: {local_loss:.1f}, Param: ",
        str(x).replace("\n", ""),
    )
    loss_list.append(local_loss)

    if return_gfs:
        return local_loss, sim_gfs
    return local_loss


def simulate_gfs(args):
    params, nwk, num_samples, ref_gfs = args
    theta, rho, gene_conv, recomb, hgt_rate = params
    try:
        theta_total_events = theta
        rho_total_events = rho * num_sites
        root_proba = theta_total_events / (rho_total_events if rho_total_events != 0 else theta)
        if not (0 <= root_proba <= 1):
            return ref_gfs * 1000

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
            double_site_relocation=True,
        )
        gm = mts.genotype_matrix(alleles=alleles)
        sim_gfs = gfs.gfs_from_matrix(gm, num_samples)
    except Exception as e:
        print(e)
        return ref_gfs * 1000
    return sim_gfs


def main(panX_sample_id):
    global loss_list
    global out_fn
    start_time = time.time()
    panX = os.path.join("panX")

    species_fn = os.path.join(panX, "species.csv")

    # nwk_fn = os.path.join(panX, "data", str(panX_sample_id), "vis", "strain_tree.nwk")
    nwk_fn = os.path.join(panX, "reduced_trees", f"reduced_{panX_sample_id}.nwk")
    gfs_fn = os.path.join(panX, "GFS", f"gfs_{panX_sample_id}.csv")

    os.makedirs(os.path.join(OUT_PATH, str(panX_sample_id)), exist_ok=True)

    out_fn = os.path.join(OUT_PATH, str(panX_sample_id), f"{prefix}fitted_params_diff_evo.json")

    plot_fn_pdf = os.path.join(OUT_PATH, str(panX_sample_id), f"{prefix}fitted_gfs.pdf")
    plot_fn_svg = os.path.join(OUT_PATH, str(panX_sample_id), f"{prefix}fitted_gfs.pdf")
    # Load Sample Properties
    with open(species_fn, "r") as f:
        lines = f.readlines()

    lines = [l.strip().split(";") for l in lines][1:]
    sample = [l for l in lines if int(l[0]) == panX_sample_id]

    if not len(sample) == 1:
        raise RuntimeError("Sample ID not found or found multiple times in species file.")

    _, species_name, number_of_samples, _, _, _ = sample[0]
    number_of_samples = int(number_of_samples)

    print("\n===============================================")
    print(f"Estimating            {species_name}\n")
    print(f"Start time:           {start_time:1.0f}")
    print(f"Number of samples:    {number_of_samples}")

    # Load True GFS
    with open(gfs_fn, "r") as f:
        lines = f.readlines()

    lines = [l.strip().split(",") for l in lines][1:]
    gene_labels_l = [l[0].strip('"') for l in lines]
    gene_labels = set(gene_labels_l)
    if not len(gene_labels) == len(gene_labels_l):
        raise RuntimeError("Inconsistency in GFS file. One or more genes appear multiple times.")

    number_of_genes = len(gene_labels)
    print(f"Number of genes       {number_of_genes}")

    # gfs_counter_full = Counter([int(l[1]) for l in lines])
    gfs_counter_reduced = Counter([int(l[2]) for l in lines])
    gfs_reduced = [gfs_counter_reduced[i] for i in range(1, number_of_samples + 1)]

    count = 0
    while gfs_reduced and gfs_reduced[-1] == 0:
        gfs_reduced.pop()
        count += 1
    reduced_num_samlpes = len(gfs_reduced)
    print(f"Reduced samples:    {reduced_num_samlpes}")
    print(f"Reduced GFS: {gfs_reduced}")
    # Load newick file
    with open(nwk_fn, "r") as f:
        nwk = f.readlines()
    nwk = nwk[0].strip()

    # Guess of initial parameters
    average_num_genes = (
        sum(gfs_counter_reduced[i] * i for i in range(1, reduced_num_samlpes + 1))
        / reduced_num_samlpes
    )
    print(f"Avg number of genes   {average_num_genes}")

    try:
        # Optimize Parameters
        theta, rho, recomb, gene_conv, hgt_rate = optim(
            average_num_genes,
            reduced_num_samlpes,
            gfs_reduced,
            nwk,
        )
    except KeyboardInterrupt:
        with open(out_fn, "r") as f:
            param = json.load(f)
            theta = param["theta"]
            rho = param["rho"]
            recomb = param["recombination"]
            gene_conv = param["gene_conversion"]
            hgt_rate = param["hgt_rate"]

    used_time = (time.time() - start_time) / 60 / 60

    print("===============================================")
    print("Optimised parameters:")
    print(f"     theta:           {theta}")
    print(f"     rho:             {rho}")
    print(f"     recombination:   {recomb}")
    print(f"     gene_conversion: {gene_conv}")
    print(f"     hgt:             {hgt_rate}")
    print(f"Writing to {out_fn}")
    print(f"Time: {used_time:.2f} [h]")
    print()

    with open(out_fn, "w") as f:
        d = {
            "theta": theta,
            "rho": rho,
            "gene_conversion": gene_conv,
            "recombination": recomb,
            "hgt_rate": hgt_rate,
            "time": used_time,
            "loss": loss_list,
        }
        d = json.dumps(d, indent=3)
        f.write(d)

    loss_list = []
    # Simulate GFS
    print("Simulating GFS")
    args = (theta, rho, gene_conv, recomb, hgt_rate)
    sim_loss, sim_gfs = loss(args, nwk, reduced_num_samlpes, gfs_reduced, return_gfs=True)

    plt.rcParams["font.family"] = "Bahnschrift"
    plt.rcParams["font.size"] = "13"
    plt.figure(figsize=(8, 4))
    plt.plot(
        range(1, len(sim_gfs) + 1),
        gfs_reduced,
        label="GFS Reduced",
        color=color_scheme.secondary,
    )
    plt.plot(
        range(1, len(sim_gfs) + 1),
        sim_gfs,
        label=f"GFS Simulated [{sim_loss:.1f}]",
        color=color_scheme.primary,
    )
    plt.xlabel("GF Class")
    plt.ylabel("Gene Frequency")
    plt.legend()
    plt.savefig(plot_fn_pdf)
    plt.savefig(plot_fn_svg)

    print("========== Done =========")


if __name__ == "__main__":
    id_list = [
        803,  # Bartonella
        # 985002,
        # 9,  # Buchnera
        # 622,
        # 1492,
    ]
    for panX_sample_id in id_list:
        main(panX_sample_id)
