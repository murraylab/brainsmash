from brainsmash.maps.core import Sampled
from brainsmash.analysis.eval import test_sampled_variogram_fits
import wbplot
# from brainsmash.neuro.cifti import export_cifti_mapping
from os.path import join
import numpy as np

# Specify paths to input/output files
data_root = "/Users/jbb/Documents/Repos/brainsmash/examples/"
output_dir = "/Users/jbb/Desktop/"


def upsample(x):
    # output = np.zeros(32492)
    # output[vertices] = x
    # return output
    output = np.zeros(59412)
    output[:29696] = x
    return output


# Read data from file
image_file = join(data_root, "cortex_left_myelin.npy")
dist_file = join(data_root, "cortex_left_distmat.npy")
index_file = join(data_root, "cortex_left_index.npy")

myelin = np.load(image_file)
distmat = np.load(dist_file, mmap_mode='r')
index = np.load(index_file, mmap_mode='r')

# Confirm visually that the simulated variograms fit well
test_sampled_variogram_fits(
    brain_map=myelin, distmat=distmat, index=index, include_naive=True)

# Compare to the variogram fits when resampling surrogate map values from the
# empirical brain map
test_sampled_variogram_fits(
    brain_map=myelin, distmat=distmat, index=index, resample=True)

# Create a few surrogate maps and plot them
generator = Sampled(brain_map=myelin, distmat=distmat,
                    index=index, resample=True)
surrogate_maps = generator(n=3)

params = {'pos-user': (1, 2.),
          'neg-user': (-2, -1),
          "disp-neg": False,
          "disp-zero": False}

for i in range(3):
    surr_map = upsample(surrogate_maps[i])
    wbplot.dscalar(join(
        output_dir, "surrogate_dense_{}.png".format(i)), surr_map,
        hemisphere='left', palette_params=params)

# # Resample from non-medial wall indices back to full surface
# cifti_map = export_cifti_mapping()['cortex_left'].to_dict()['vertex']
# vertices = np.sort(list(cifti_map.values()))
# nv = vertices.size
# assert nv == 29696
