from brainsmash.geo.dists import cortex, parcellate, subcortex
from time import time

# Compute dense geodesic distance matrix
surf = "/Users/jbb/Documents/Repos/brainsmash/brainsmash/data/surfaces/"
surf += "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
outfile = "/Users/jbb/Documents/Repos/brainsmash/examples/test_geo_dists.txt"
of = cortex(surface=surf, outfile=outfile, euclid=False)

# Compute parcellated geodesic distance matrix
ofp = "/Users/jbb/Documents/Repos/brainsmash/examples/"
ofp += "test_geo_dists_parcel.txt"
dlbl = "/Users/jbb/Documents/Repos/brainsmash/brainsmash/data/L_Q1-Q6_Related"
dlbl += "Validation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors."
dlbl += "32k_fs_LR.label.gii"
parcellate(of, dlbl, ofp, delimiter=" ")

# Compute dense Euclidean distance matrix
eoutfile = "/Users/jbb/Documents/Repos/brainsmash/examples/"
eoutfile += "test_euclid_dists.txt"
t1 = time()
ofe = cortex(surface=surf, outfile=eoutfile, euclid=True)
print(time() - t1)

# Compute parcellated Euclidean distance matrix
ofpe = "/Users/jbb/Documents/Repos/brainsmash/examples/"
ofpe += "test_euclid_dists_parcel.txt"
t1 = time()
parcellate(ofe, dlbl, ofpe)
print(time() - t1)

# Compute dense Euclidean distance matrix for subcortex
soutfile = "/Users/jbb/Documents/Repos/brainsmash/examples/test_subcortex.txt"
t1 = time()
ofs = subcortex(soutfile)
print(time()-t1)

# Create memory-mapped arrays from new distance matrices
