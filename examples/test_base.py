from brainsmash.nulls.core import Base
from brainsmash.neuro.io import load_cifti2
from brainsmash.analysis.eval import test_variogram_fits
import wbplot
from os.path import join
import numpy as np

# Specify paths to data files
root = "/Users/jbb/Documents/Repos/brainsmash/examples/"
image = join(root, "myelin_L.pscalar.nii")
matrix = join(root, "left_parcel_distmat.npy")

# Read data from file
myelin = load_cifti2(image)
distmat = np.load(matrix)

# Instantiate Base class
generator = Base(brain_map=myelin, distmat=distmat)

# Confirm visually that the simulated variograms fit well
test_variogram_fits(brain_map=myelin, distmat=distmat)

# Compare to the variogram fits when resampling surrogate map values from the
# empirical brain map
test_variogram_fits(brain_map=myelin, distmat=distmat, resample=True)

