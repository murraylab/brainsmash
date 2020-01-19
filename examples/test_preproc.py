from brainsmash.geo.dists import cortex

# Compute dense geodesic distance matrix
surf = "/Users/jbb/Documents/Repos/brainsmash/brainsmash/data/surfaces/"
surf += "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"

# Compute parcellated geodesic distance matrix
outfile = "/Users/jbb/Documents/Repos/brainsmash/examples/test_geo_dists.txt"
of = cortex(surface=surf, outfile=outfile, euclid=False)

# Compute dense Euclidean distance matrix

# Compute parcellated Euclidean distance matrix

# Create memory-mapped arrays from new distance matrices
