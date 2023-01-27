nx=33

# No normalization for qpsi! Instead, code normalizes/denormalizes w/ inverse
#   i.e. by transforming to iota = 1/q (mean & std for q would be ignored)
normalizations={
    'zipfit_etempfit_psi': {'mean': 0, 'std': 1},
    'zipfit_edensfit_psi': {'mean': 0, 'std': 1},
    'zipfit_trotfit_psi': {'mean': 0, 'std': 1e2},
    'zipfit_itempfit_psi': {'mean': 0, 'std': 1},
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
