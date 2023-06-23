import torch
import numpy as np
import copy
import os
import h5py as h5
import matplotlib.pyplot as plt
import orso_nn_helpers

def get_sig(dic,key):
    name_map={'a':'aminor_EFIT01','betan':'betan_EFIT01','bt':'bt','ip':'ip','kappa':'kappa_EFIT01','r':'rmaxis_EFIT01','neped':'neped_joe'}
    if key in name_map:
        ret=dic[name_map[key]][()]
        if key=='ip':
            ret*=1e-6
        return ret
    elif key=='delta':
        return (dic['tritop_EFIT01'][()]+dic['tribot_EFIT01'][()])/2
    # elif key=='neped':
    #     num_rho_points=dic['zipfit_edensfit_rho'].shape[-1]
    #     return dic['zipfit_edensfit_rho'][:,int(num_rho_points*0.8)]
    elif key=='zeffped':
        return 2*np.ones_like(dic['ip'][()])
    elif key=='m':
        return 2*np.ones_like(dic['ip'][()])
    else:
        return np.nan*np.ones_like(dic['ip'][()])

modelDir='multiroot'
ensemble_info=orso_nn_helpers.get_ensemble_info(modelDir)
filename='test.h5' #aug_data
#test_densities=[1,3,5,7]
with h5.File(filename,'a') as f:
    shots = list(f.keys())
    shots.remove('times')
    shots.remove('spatial_coordinates')
    for shot in shots:
        # add neped estimate
        if 'zipfit_edensfit_rho' in f[shot]:
            if 'neped_joe' in f[shot]:
                del f[shot]['neped_joe']
            rho_ind=26
            f[shot]['neped_joe']=f[shot]['zipfit_edensfit_rho'][:,rho_ind]
        ensemble_means=[]
        inputs=[]
        try:
            for input_name in ensemble_info['input_names']:
                inputs.append(get_sig(f[shot],input_name))
            all_sigs_available=True
        except:
            all_sigs_available=False
        if all_sigs_available:
            inputs=np.array(inputs).T
            ensemble_means=np.zeros(len(inputs))
            for input_ind, test_input in enumerate(inputs):
                ensemble_mean,ensemble_std=orso_nn_helpers.evaluate_model(test_input,ensemble_info)
                ensemble_means[input_ind]=ensemble_mean[0]
            if 'epedHeight' in f[shot]:
                del f[shot]['epedHeight']
            f[shot]['epedHeight']=ensemble_means
            if 'eped_te_prediction' in f[shot]:
                del f[shot]['eped_te_prediction']
            # unit cnonversion from OMFIT's EPED module scripts
            # *1e3, /1.6e-19, *1e19, /2 (the 2 is for electron/ion split I think)
            f[shot]['eped_te_prediction']=f[shot]['epedHeight'][:] / f[shot]['neped_joe'][:] *1e3/1.6/2

        else:
            print(f'{shot} eped failed')
