"""
Default parameters required for the custom simulation and mutation model.
"""

from numpy import inf


class Args:
    def __init__(
        self,
        sample_size,
        num_sites,
        gene_conversion_rate,
        recombination_rate,
        hgt_rate,
        ce_from_ts,
        ce_from_nwk,
        random_seed,
    ) -> None:

        # Default / unused parameters
        self.output_file = None
        self.verbose = False
        self.log_level = 0
        self.discrete = True
        self.num_replicates = 1000
        self.recomb_positions = None
        self.recomb_rates = None
        self.num_populations = 1
        self.migration_rate = 0
        self.sample_configuration = None
        self.population_growth_rates = None
        self.population_sizes = None
        self.population_size_change = []
        self.population_growth_rate_change = []
        self.migration_matrix_element_change = []
        self.bottleneck = []
        self.census_time = []
        self.trajectory = None
        self.all_segments = False
        self.additional_nodes = 0
        self.time_slice = 1e-06
        self.model = "hudson"
        self.from_ts = None
        self.end_time = inf
        self.svg = False
        self.d3 = False
        self.ce_from_ts = None
        self.ce_from_nwk = None
        # Adaptable parameters
        self.sample_size = sample_size
        self.sequence_length = num_sites
        self.gene_conversion_rate = [gene_conversion_rate, 1.0]
        self.recombination_rate = recombination_rate
        self.hgt_rate = hgt_rate
        self.ce_from_ts = ce_from_ts
        self.ce_from_nwk = ce_from_nwk
        self.random_seed = random_seed
