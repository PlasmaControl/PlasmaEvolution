import numpy as np
import copy
import torch
# number of x points in profile data
nx=33
# timestep in dataset, in seconds
DT=0.02

# No normalization for qpsi! Instead, code normalizes/denormalizes w/ inverse
#   i.e. by transforming to iota = 1/q (mean & std for q would be ignored)
normalizations={
    'zipfit_etempfit_rho': {'mean': 0, 'std': 2},
    'zipfit_edensfit_rho': {'mean': 0, 'std': 2},
    'neped_joe': {'mean': 0, 'std': 1},
    'zipfit_trotfit_rho': {'mean': 0, 'std': 1e2},
    'zipfit_itempfit_rho': {'mean': 0, 'std': 2},
    'zipfit_zdensfit_rho': {'mean': 0, 'std': 2},
    'pres_EFIT01': {'mean': 0, 'std': 1e4},
    'pinj': {'mean': 0, 'std': 2e3},
    'tinj': {'mean': 0, 'std': 2},
    'ipsiptargt': {'mean': 0, 'std': 1},
    'ip': {'mean': 0, 'std': 1e6},
    'bt': {'mean': 0, 'std': 1},
    'dstdenp': {'mean': 0, 'std': 2},
    'gasA': {'mean': 0, 'std': 1},
    'gasB': {'mean': 0, 'std': 1},
    'gasC': {'mean': 0, 'std': 1},
    'gasD': {'mean': 0, 'std': 1},
    'li_EFIT01': {'mean': 0, 'std': 1},
    'tribot_EFIT01': {'mean': 0, 'std': 1},
    'tritop_EFIT01': {'mean': 0, 'std': 1},
    'aminor_EFIT01': {'mean': 0, 'std': 1},
    'rmaxis_EFIT01': {'mean': 0, 'std': 1},
    'dssdenest': {'mean': 0, 'std': 2},
    'kappa_EFIT01': {'mean': 0, 'std': 1},
    'volume_EFIT01': {'mean': 0, 'std': 10},
    'betan_EFIT01': {'mean': 0, 'std': 1},
    'epedHeight': {'mean': 0, 'std': 5e-3},
    'eped_te_prediction': {'mean': 0, 'std': 1},
    'epedHeightForNe1': {'mean': 0, 'std': 5e-3},
    'epedHeightForNe3': {'mean': 0, 'std': 5e-3},
    'epedHeightForNe5': {'mean': 0, 'std': 5e-3},
    'epedHeightForNe7': {'mean': 0, 'std': 5e-3},
    'D_tot': {'mean': 0, 'std': 1e2},
    'H_tot': {'mean': 0, 'std': 1e2},
    'Ar_tot': {'mean': 0, 'std': 1e2},
    'Ne_tot': {'mean': 0, 'std': 1e2},
    'He_tot': {'mean': 0, 'std': 1e2},
    'N_tot': {'mean': 0, 'std': 1e2},
    'ech_pwr_total': {'mean': 0, 'std': 1e6},
    'P_AUXILIARY': {'mean': 0, 'std': 2e3},       # custom signals,
    'zeff_rho': {'mean': 0, 'std': 2}, # defined in customDatasetMakers ad hoc
    }
clipped_signals={}

# add ASTRA stuff, for all possible ASTRA runs
sig_normalizations={
    'CD': {'mean': 0, 'std': 1},
    'CC': {'mean': 0, 'std': 50},
    'CUBS': {'mean': 0, 'std': 1},
    'HE': {'mean': 0, 'std': 1},
    'XI': {'mean': 0, 'std': 1},
    'PITOT': {'mean': 0, 'std': 2},
    'PIBM': {'mean': 0, 'std': 2},
    'PETOT': {'mean': 0, 'std': 2},
    'PEBM': {'mean': 0, 'std': 2},
    'TE': {'mean': 0, 'std': 1},
    'TI': {'mean': 0, 'std': 1},
    'NI': {'mean': 0, 'std': 2},
    'ANGF': {'mean': 0, 'std': 1e2},
    'UPAR': {'mean': 0, 'std': 1e2},
    'NE': {'mean': 0, 'std': 1}
}
sig_bounds={
    'HE': {'min': 0, 'max': 20},
    'XI': {'min': 0, 'max': 20}
}
for astrasim in ['astrainterpretive','astrapredictEPEDNNTGLFNNfullyZIPFIT',
                 'astrainterpretZIPFIT', 'astrapredictTGLFNNZIPFIT']:
    for sig in sig_normalizations:
        normalizations[f'{sig}_{astrasim}']=sig_normalizations[sig]
    for sig in sig_bounds:
        clipped_signals[f'{sig}_{astrasim}']=sig_bounds[sig]

# if use_gyroBohm:
#     normalizations['zipfit_edensfit_rho'] = {'mean': 0, 'std': 5e-6}
#     #normalizations['zipfit_etempfit_rho'] = {'mean': 0, 'std': 1}
#     normalizations['zipfit_itempfit_rho'] = {'mean': 0, 'std': 1}

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

# def get_gyro_normalized_dic(input_dic):
#     output_dic = copy.copy(input_dic)
#     #Te = np.clip(input_dic['zipfit_etempfit_rho'], 0.01, None)
#     Ti = np.clip(input_dic['zipfit_itempfit_rho'], 0.01, None)
#     ne = np.clip(input_dic['zipfit_edensfit_rho'], 0.01, None)
#     Bt = np.abs(np.repeat(input_dic['bt'][:,:,np.newaxis], len(Ti[0][0]), axis=2)) # make Bt flat radial profile
#     a = np.repeat(input_dic['aminor_EFIT01'][:,:,np.newaxis], len(Ti[0][0]), axis=2) # make a flat radial profile
#     Ip = np.abs(np.repeat(input_dic['ip'][:,:,np.newaxis], len(Ti[0][0]), axis=2))
#     f_gr = ne / (Ip/(np.pi*a**2))
#     rho_star = np.sqrt(Ti) / (a * Bt)
#     output_dic['zipfit_itempfit_rho']=rho_star
#     output_dic['zipfit_edensfit_rho']=f_gr
#     return output_dic

# def get_gyro_denormalized_dic(input_dic):
#     output_dic = copy.copy(input_dic)
#     #temp_frac = input_dic['zipfit_etempfit_rho']
#     rho_star = input_dic['zipfit_itempfit_rho']
#     f_gr = input_dic['zipfit_edensfit_rho']
#     #Bt = np.abs(np.repeat(input_dic['bt'][:,:,np.newaxis], len(beta[0][0]), axis=2)) # make Bt flat radial profile
#     #a = np.repeat(input_dic['aminor_EFIT01'][:,:,np.newaxis], len(beta[0][0]), axis=2)
#     Bt = 2
#     a = 0.5
#     Ip = 1000000
#     Ti = (a * Bt * rho_star)**2
#     #Te = temp_frac * Ti
#     ne = f_gr * (Ip/(np.pi*a**2))
#     #output_dic['zipfit_etempfit_rho']=Te
#     output_dic['zipfit_itempfit_rho']=Ti
#     output_dic['zipfit_edensfit_rho']=ne
#     return output_dic

def get_rotation_sigs(sigs):
    rotation_signals=[]
    for sig in sigs:
        if sig=='zipfit_trotfit_rho' or sig.startswith('UPAR_'):
            rotation_signals.append(sig)
    return rotation_signals
def get_density_sigs(sigs):
    density_signals=[]
    for sig in sigs:
        if sig=='zipfit_edensfit_rho' or sig.startswith('NE_'):
            density_signals.append(sig)
    return density_signals

# excluded_sigs for e.g. shotnum and times from preprocessed data
# assumes dictionary of signals, each of form [...,num_rho] / [...]
# e.g. (rho) / scalar; (time, rho) / (time); or (nsamples, time, rho) / (nsamples, time)
def get_normalized_dic(denormed_dic, excluded_sigs=['shotnum','times'], use_fancy_normalization=False):
    for sig in denormed_dic:
        denormed_dic[sig]=np.array(denormed_dic[sig])
    normed_dic={}
    excluded_sigs=[sig for sig in denormed_dic.keys() if sig in excluded_sigs]
    for sig in excluded_sigs:
        normed_dic[sig]=denormed_dic[sig]
    considered_sigs=[sig for sig in denormed_dic.keys() if sig not in excluded_sigs]
    if use_fancy_normalization:
        gyrobohm_rotation_signals=get_rotation_sigs(denormed_dic.keys())
        gyrobohm_density_signals=get_density_sigs(denormed_dic.keys())
        for sig in gyrobohm_rotation_signals+gyrobohm_density_signals+['pinj']:
            if sig in denormed_dic:
                considered_sigs.remove(sig)
    if use_fancy_normalization:
        density_sig='zipfit_edensfit_rho'
        volume_sig='volume_EFIT01'
        r_sig='rmaxis_EFIT01'
        a_sig='aminor_EFIT01'
        ip_sig='ip'
        # use volume average power
        if 'pinj' in denormed_dic:
            normed_dic['pinj']=(denormed_dic['pinj']/normalizations['pinj']['std']) / (denormed_dic[volume_sig]/normalizations[volume_sig]['std'])
        for sig in gyrobohm_density_signals:
            # ip can be negative, density is always positive so only need the abs in the normalization and not denormalization
            greenwald_density=(np.abs(denormed_dic[ip_sig])/normalizations[ip_sig]['std']) / (denormed_dic[a_sig]/normalizations[a_sig]['std'])**2
            normed_dic[sig]=(denormed_dic[sig]/normalizations[sig]['std'])/ greenwald_density[...,None]
        for sig in gyrobohm_rotation_signals:
            num_rho=denormed_dic[sig].shape[-1]
            rho=np.linspace(0,1,num_rho)
            # convert upar to momentum~upar*M*R^2~upar*ne*Vol*R^2
            #    since ne~sum(Z_i n_i) by quasineutrality and Z_i~mass (though complicated for partially ionized Tungsten)
            #    and to avoid division by 0 take integral of ne*Vol, which is like sum(rho * ne)*V since volume of elemnts scales like rho
            mass=np.mean(rho* (denormed_dic[density_sig]/normalizations[density_sig]['std']), axis=-1)* (denormed_dic[volume_sig]/normalizations[volume_sig]['std'])
            moment_of_inertia=mass[...,None]* (denormed_dic[r_sig][...,None]/normalizations[r_sig]['std'])**2
            normed_dic[sig]=denormed_dic[sig]/normalizations[sig]['std']* moment_of_inertia
    for sig in considered_sigs:
        if 'qpsi' in sig:
            normed_dic[sig] = 1. / denormed_dic[sig]
        else:
            normed_dic[sig] = (denormed_dic[sig] - normalizations[sig]['mean']) / normalizations[sig]['std']
    return normed_dic

def get_denormalized_dic(normed_dic, excluded_sigs=['shotnum','times'], use_fancy_normalization=False):
    for sig in normed_dic:
        normed_dic[sig]=np.array(normed_dic[sig])
    denormed_dic={}
    excluded_sigs=[sig for sig in normed_dic.keys() if sig in excluded_sigs]
    for sig in excluded_sigs:
        denormed_dic[sig]=normed_dic[sig]
    considered_sigs=[sig for sig in normed_dic.keys() if sig not in excluded_sigs]
    if use_fancy_normalization:
        gyrobohm_rotation_signals=get_rotation_sigs(normed_dic.keys())
        gyrobohm_density_signals=get_density_sigs(normed_dic.keys())
        for sig in gyrobohm_rotation_signals+gyrobohm_density_signals+['pinj']:
            if sig in normed_dic:
                considered_sigs.remove(sig)
    for sig in considered_sigs:
        if 'qpsi' in sig:
            denormed_dic[sig] = 1. / normed_dic[sig]
        else:
            denormed_dic[sig] = (normed_dic[sig] * normalizations[sig]['std']) + normalizations[sig]['mean']
    if use_fancy_normalization:
        density_sig='zipfit_edensfit_rho'
        volume_sig='volume_EFIT01'
        r_sig='rmaxis_EFIT01'
        a_sig='aminor_EFIT01'
        ip_sig='ip'
        if 'pinj' in normed_dic:
            denormed_dic['pinj']=normed_dic['pinj']*(denormed_dic[volume_sig]/normalizations[volume_sig]['std'])* normalizations['pinj']['std']
        for sig in gyrobohm_density_signals:
            greenwald_density=(denormed_dic[ip_sig]/normalizations[ip_sig]['std']) / (denormed_dic[a_sig]/normalizations[a_sig]['std'])**2
            denormed_dic[sig]=(normed_dic[sig]*greenwald_density[...,None])* normalizations[sig]['std']
        for sig in gyrobohm_rotation_signals:
            num_rho=normed_dic[sig].shape[-1]
            rho=np.linspace(0,1,num_rho)
            mass=np.mean(rho* (denormed_dic[density_sig]/normalizations[density_sig]['std']) ,axis=-1)* (denormed_dic[volume_sig]/normalizations[volume_sig]['std'])
            moment_of_inertia=mass[...,None]*denormed_dic[r_sig][...,None]**2/normalizations[r_sig]['std']**2
            denormed_dic[sig]=normalizations[sig]['std']*normed_dic[sig] / moment_of_inertia
    return denormed_dic
