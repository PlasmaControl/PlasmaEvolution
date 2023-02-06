# number of x points in profile data
nx=33
# timestep in dataset, in seconds
DT=0.025

# No normalization for qpsi! Instead, code normalizes/denormalizes w/ inverse
#   i.e. by transforming to iota = 1/q (mean & std for q would be ignored)
normalizations={
    'zipfit_etempfit_rho': {'mean': 0, 'std': 1},
    'zipfit_edensfit_rho': {'mean': 0, 'std': 1},
    'zipfit_trotfit_rho': {'mean': 0, 'std': 1e2},
    'zipfit_itempfit_rho': {'mean': 0, 'std': 1},
    'pres_EFIT01': {'mean': 0, 'std': 1e4},
    'pinj': {'mean': 0, 'std': 1e3},
    'tinj': {'mean': 0, 'std': 1},
    'ipsiptargt': {'mean': 0, 'std': 1},
    'bt': {'mean': 0, 'std': 1},
    'dstdenp': {'mean': 0, 'std': 1},
    'li_EFIT01': {'mean': 0, 'std': 1},
    'tribot_EFIT01': {'mean': 0, 'std': 1},
    'tritop_EFIT01': {'mean': 0, 'std': 1},
    'dssdenest': {'mean': 0, 'std': 1},
    'kappa_EFIT01': {'mean': 0, 'std': 1},
    'volume_EFIT01': {'mean': 0, 'std': 10}
    }

min_shot=170010
max_shot=170100
val_indices=[5]
test_indices=[0]

train_shots=[shot for shot in range(min_shot,max_shot) if shot%10 not in val_indices+test_indices]
val_shots=[shot for shot in range(min_shot,max_shot) if shot%10 in val_indices]
test_shots=[shot for shot in range(min_shot,max_shot) if shot%10 in test_indices]

# ohmic power in Watts, to add to Pinj to get power for taue calculation
ohmicPower=5e5
# min and max taue in seconds
taueMin=0.010
taueMax=1.000

IMPURITY_FRACTION=0.04
# or from
#Zeff=2 flat profile
#Zimp=6
#IMPURITY_FRACTION=(Zeff-1)/(Zimp*(Zimp-Zeff)) #=1/24~4%
IMPURITY_Z=6
KEV_PER_1019_TO_J=1.602e3

def normalize(arr, sig_name):
    # q blows up at the edge, use iota = 1/q as proxy for q and don't use ad hoc normalization
    if 'qpsi' in sig_name:
        normed_arr = 1. / arr
    else:
        normed_arr = (arr - normalizations[sig_name]['mean']) / normalizations[sig_name]['std']
    return normed_arr
def denormalize(arr, sig_name):
    if 'qpsi' in sig_name:
        denormed_arr = 1. / arr
    else:
        denormed_arr = (arr * normalizations[sig_name]['std']) + normalizations[sig_name]['mean']
    return denormed_arr
