from brainsmash.maps.core import Smash
from brainsmash.analysis.eval import test_smash_sampled_variogram_fits
from brainsmash.utils import preproc
from brainsmash.neuro.io import export_cifti_mapping
import wbplot
from pathlib import Path
import numpy as np

# TODO build some of this framework into smash to handle possible errors?

# Specify paths to input/output files
data_root = "/Users/jbb/Documents/Repos/brainsmash/examples/"
outdir = "/Users/jbb/Desktop/"
df = Path("/Users/jbb/Documents/Research/projects/surrogates/outputs/")
df = df / "Geodesic_distance_dense.txt"
image_nii_file = str(Path(data_root) / "myelin_L.dscalar.nii")
image_txt_file = str(Path(data_root) / "myelin_L.txt")

# Create medial wall mask
surface_indices = export_cifti_mapping()['cortex_left'].values.squeeze()
mask = np.ones(32492, dtype=np.int32)
mask[surface_indices] = 0
mask_file = str(Path(data_root) / 'medial_mask.txt')
np.savetxt(fname=mask_file, X=mask, fmt='%i')


def upsample(x):
    # output = np.zeros(32492)
    # output[vertices] = x
    # return output
    output = np.zeros(59412)
    output[:29696] = x
    return output


# Run preprocessing steps
preproc.image2txt(image_nii_file, outfile=image_txt_file, maskfile=mask_file)
# fnames = preproc.txt2mmap(
#     dist_file=df, output_dir=data_root, maskfile=mask_file, delimiter=" ")
fnames = {'distmat': '/Users/jbb/Documents/Repos/brainsmash/examples/distmat.npy',
          'index': '/Users/jbb/Documents/Repos/brainsmash/examples/index.npy'}

# # Create a few surrogate maps and plot them
# generator = Smash(
#     brain_map=image_txt_file, distmat=fnames['distmat'],
#     index=fnames['index'], resample=False)
#
# surrogate_maps = generator(n=3)
#
# params = {'pos-percent': (2, 98),
#           'neg-percent': (2, 98),
#           "disp-neg": False,
#           "disp-zero": False}
#
# for i in range(3):
#     surr_map = upsample(surrogate_maps[i])
#     fout = str(Path(outdir) / "surrogate_dense_{}.png".format(i))
#     wbplot.dscalar(fout, surr_map, hemisphere='left', palette_params=params)

# Confirm visually that the simulated variograms fit well
distmat = fnames['distmat']
index = fnames['index']
test_smash_sampled_variogram_fits(
    brain_map=image_txt_file, distmat=distmat, index=index,
    include_naive=True, nsurr=5)

# Compare to the variogram fits when resampling surrogate map values from the
# empirical brain map
test_smash_sampled_variogram_fits(
    brain_map=image_txt_file, distmat=distmat, index=index, include_naive=True,
    resample=True)
