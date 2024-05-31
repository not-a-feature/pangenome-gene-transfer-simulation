"""
Tests the impact of double gene gain events for different ratios of theta / rho.
"""

import gene_model

import numpy as np
import itertools
from multiprocessing import Pool
from tqdm import tqdm
import os

SELF_FN = os.path.dirname(os.path.abspath(__file__))
OUT_FN = os.path.join(SELF_FN, "ratio_percent.csv")

num_sites = np.arange(1000, 10000, 500)

theta = np.arange(1000, 10500, 500)
rho = np.arange(0.1, 1.05, 0.05)

kwagrs = {"num_samples": 10, "gene_conversion_rate": 0}

num_simulations = 100
max_workers = 14


def single_run(kwagrs):
    try:
        count = gene_model.gene_model(**kwagrs)
    except Exception as e:
        # print(e, kwagrs)
        return (-1, -1)

    return count


with open(OUT_FN, "w") as f:
    f.write(f"num_sites,theta,rho,")
    f.write(",".join([f"d{i},t{i}" for i in range(num_simulations)]))
    f.write("\n")


combinations = list(itertools.product(theta, rho))
combinations = list(itertools.product(num_sites, combinations))
# np.random.shuffle(combinations)


combinations = [(gc, (t, r)) for gc, (t, r) in combinations if t / r <= gc]
print(f"Testing {len(combinations)} combinations")

for gc, (t, r) in tqdm(combinations):
    kwagrs["num_sites"] = gc
    kwagrs["theta"] = t
    kwagrs["rho"] = r

    kwagrs_list = [kwagrs] * num_simulations
    with Pool(processes=max_workers) as pool:
        result = pool.imap_unordered(single_run, kwagrs_list)
        measurements = list(result)
        measurements = ",".join([f"{i},{j}" for i, j in measurements])

    with open(OUT_FN, "a") as f:
        f.write(f"{gc},{t},{r},{measurements}")
        f.write("\n")
