import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

primary = "#5dc694"
secondary = "#204e66"
blue = "#2398ff"
red = "#d72d47"
background = "#d2dbe0"


def get_cmap(secondary, primary, min_offset, max_offset):
    """Linearly interpolates between two colors in RGB space."""
    primary_rgb = to_rgb(primary)
    secondary_rgb = to_rgb(secondary)

    r = (max(secondary_rgb[0] - min_offset, 0), min(primary_rgb[0] + max_offset, 1))
    g = (max(secondary_rgb[1] - min_offset, 0), min(primary_rgb[1] + max_offset, 1))
    b = (max(secondary_rgb[2] - min_offset, 0), min(primary_rgb[2] + max_offset, 1))
    interpolated_colors = np.stack((r, g, b), axis=1)

    cmap_name = "tskit_cmap"
    return LinearSegmentedColormap.from_list(cmap_name, interpolated_colors)


cmap = get_cmap(secondary, primary, 0.1, 0.1)
