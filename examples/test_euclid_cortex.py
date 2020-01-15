from brainsmash.geo.dists import compute_cortex_dists
from time import time

surf_left = ("/Users/jbb/Documents/Repos/brainsmash/brainsmash/data/surfaces/"
             "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii")
surf_right = ("/Users/jbb/Documents/Repos/brainsmash/brainsmash/data/surfaces/"
              "S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii")

t1 = time()
compute_cortex_dists(
    surface=surf_left,
    fout="/Users/jbb/Desktop/left_cortex_euclid_test",
    euclid=True)
print(time() - t1)
