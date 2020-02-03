from os.path import join, dirname, abspath

# TODO document this file!

# TODO __all__ = []

kernels = ['exp', 'gaussian', 'invdist', 'uniform']

repo_root = dirname(dirname(abspath(__file__)))  # root directory path
package_root = join(repo_root, "brainsmash")
data = join(package_root, "data")  # data directory
surfaces = join(data, "surfaces")  # surface files

parcel_labels_lr = join(
    data, "CortexSubcortex_ColeAnticevic_NetPartition_"
          "wSubcorGSR_netassignments_LR.dlabel.nii")
