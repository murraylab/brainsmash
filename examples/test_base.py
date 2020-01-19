from brainsmash.maps.core import Base
from brainsmash.neuro.io import load_data
from brainsmash.analysis.eval import test_base_variogram_fits
import wbplot
from os.path import join
import numpy as np

# Specify paths to input/output files
data_root = "/Users/jbb/Documents/Repos/brainsmash/examples/"
output_dir = "/Users/jbb/Desktop/"

# Read data from file
image = join(data_root, "myelin_L.pscalar.nii")
# image = join(data_root, "myelin_L_parcel.txt")
matrix = join(data_root, "left_parcel_distmat.npy")

myelin = load_data(image)
distmat = np.load(matrix)

# # Confirm visually that the simulated variograms fit well
# test_base_variogram_fits(brain_map=myelin, distmat=distmat, include_naive=True)
#
# # Compare to the variogram fits when resampling surrogate map values from the
# # empirical brain map
# test_base_variogram_fits(brain_map=myelin, distmat=distmat, resample=True)


# Create a few surrogate maps and plot them
# generator = Base(brain_map=myelin, distmat=distmat, resample=True)
generator = Base(brain_map=myelin, distmat=distmat, resample=True)

surrogate_maps = generator(n=2)
for i in range(surrogate_maps.shape[1]):
    wbplot.pscalar(join(
        output_dir, "surrogate_{}.png".format(i)), surrogate_maps[i],
        hemisphere='left', vrange=(1.1, 1.5))
