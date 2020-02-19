from os.path import join, dirname, abspath

__all__ = ['kernels', 'parcel_labels_lr']

# Names of available kernels, which are defined in mapgen.kernels.py
kernels = ['exp', 'gaussian', 'invdist', 'uniform']

repo_root = dirname(dirname(abspath(__file__)))  # root directory path
package_root = join(repo_root, "brainsmash")  # package directory
data = join(package_root, "data")  # data directory

# This file is used by default to identify MNI coordinates of each subcortical
# voxel, and to establish the mapping between CIFTI indices and different gross
# anatomical structures
parcel_labels_lr = join(
    data, "CortexSubcortex_ColeAnticevic_NetPartition_"
          "wSubcorGSR_netassignments_LR.dlabel.nii")
