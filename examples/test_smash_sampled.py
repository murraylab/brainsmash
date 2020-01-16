from brainsmash.maps.core import Smash
from brainsmash.utils import preproc
import wbplot
from pathlib import Path
import numpy as np

# Specify paths to input/output files
data_root = "/Users/jbb/Documents/Repos/brainsmash/examples/"
df = Path("/Users/jbb/Documents/Research/projects/surrogates/outputs/")
df = df / "Geodesic_distance_dense.txt"
xf = Path(data_root) /


def upsample(x):
    # output = np.zeros(32492)
    # output[vertices] = x
    # return output
    output = np.zeros(59412)
    output[:29696] = x
    return output


# Run preprocessing steps
fnames = preproc.txt2mmap(
    dist_file=df, output_dir=data_root, maskfile=None, delimiter=" ")
preproc.image2txt(image_file, )
# Create a few surrogate maps and plot them
generator = Smash(brain_map=myelin, distmat=distmat,
                  index=index, resample=False)
surrogate_maps = generator(n=3)

params = {'pos-percent': (2, 98),
          'neg-percent': (2, 98),
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
