from os.path import join, dirname, abspath

kernels = ['exp', 'gaussian', 'invdist', 'uniform']

root = dirname(dirname(abspath(__file__)))  # root directory path
root = join(root, "brainsmash")
data = join(root, "data")  # data directory
surfaces = join(data, "surfaces")

# -------------
# Surface files
# -------------

ctx_left_surface = join(
    surfaces, "S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii")
ctx_right_surface = join(
    surfaces, "S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii")
parcel_labels_lr = join(
    data, "CortexSubcortex_ColeAnticevic_NetPartition_"
          "wSubcorGSR_netassignments_LR.dlabel.nii")
scene = join(data, "Human.scene")
