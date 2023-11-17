import numpy as np
import copy
import torch
# number of x points in profile data
nx=33
# timestep in dataset, in seconds
DT=0.02
use_gyroBohm = False

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
    'ech_pwr_total': {'mean': 0, 'std': 1e6}
    }
clipped_signals={}

# add ASTRA stuff, for all possible ASTRA runs
sig_normalizations={
    'CC': {'mean': 0, 'std': 50},
    'CUBS': {'mean': 0, 'std': 1},
    'HE': {'mean': 0, 'std': 1},
    'XI': {'mean': 0, 'std': 1},
    'PITOT': {'mean': 0, 'std': 2},
    'PIBM': {'mean': 0, 'std': 2},
    'PETOT': {'mean': 0, 'std': 2},
    'PEBM': {'mean': 0, 'std': 2}
}
sig_bounds={
    'HE': {'min': 0, 'max': 20},
    'XI': {'min': 0, 'max': 20}
}
for astrasim in ['astrainterpretiveZIPFIT', 'astrainterpretiveECHZIPFIT']:
    for sig in sig_normalizations:
        normalizations[f'{sig}_{astrasim}']=sig_normalizations[sig]
    for sig in sig_bounds:
        clipped_signals[f'{sig}_{astrasim}']=sig_bounds[sig]

if use_gyroBohm:
    normalizations['zipfit_edensfit_rho'] = {'mean': 0, 'std': 5e-6}
    #normalizations['zipfit_etempfit_rho'] = {'mean': 0, 'std': 1}
    normalizations['zipfit_itempfit_rho'] = {'mean': 0, 'std': 1}
# if average normalized data for shot greater than this many deviations away,
# exclude the shot from the dataset
deviation_cutoff=10

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

def get_gyro_normalized_dic(input_dic):
    output_dic = copy.copy(input_dic)
    #Te = np.clip(input_dic['zipfit_etempfit_rho'], 0.01, None)
    Ti = np.clip(input_dic['zipfit_itempfit_rho'], 0.01, None)
    ne = np.clip(input_dic['zipfit_edensfit_rho'], 0.01, None)
    Bt = np.abs(np.repeat(input_dic['bt'][:,:,np.newaxis], len(Ti[0][0]), axis=2)) # make Bt flat radial profile
    a = np.repeat(input_dic['aminor_EFIT01'][:,:,np.newaxis], len(Ti[0][0]), axis=2) # make a flat radial profile
    Ip = np.abs(np.repeat(input_dic['ip'][:,:,np.newaxis], len(Ti[0][0]), axis=2))
    f_gr = ne / (Ip/(np.pi*a**2))
    rho_star = np.sqrt(Ti) / (a * Bt)
    output_dic['zipfit_itempfit_rho']=rho_star
    output_dic['zipfit_edensfit_rho']=f_gr
    return output_dic

def get_gyro_denormalized_dic(input_dic):
    output_dic = copy.copy(input_dic)
    #temp_frac = input_dic['zipfit_etempfit_rho']
    rho_star = input_dic['zipfit_itempfit_rho']
    f_gr = input_dic['zipfit_edensfit_rho']
    #Bt = np.abs(np.repeat(input_dic['bt'][:,:,np.newaxis], len(beta[0][0]), axis=2)) # make Bt flat radial profile
    #a = np.repeat(input_dic['aminor_EFIT01'][:,:,np.newaxis], len(beta[0][0]), axis=2)
    Bt = 2
    a = 0.5
    Ip = 1000000
    Ti = (a * Bt * rho_star)**2
    #Te = temp_frac * Ti
    ne = f_gr * (Ip/(np.pi*a**2))
    #output_dic['zipfit_etempfit_rho']=Te
    output_dic['zipfit_itempfit_rho']=Ti
    output_dic['zipfit_edensfit_rho']=ne
    return output_dic

# excluded_sigs for e.g. shotnum and times from preprocessed data
def get_normalized_dic(denormed_dic, excluded_sigs=[]):
    if use_gyroBohm:
        denormed_dic=get_gyro_normalized_dic(denormed_dic)
    normalized_dic={}
    for sig in denormed_dic:
        denormed_dic[sig]=np.array(denormed_dic[sig])
        if sig not in excluded_sigs:
            if 'qpsi' in sig:
                normalized_dic[sig] = 1. / denormed_dic[sig]
            else:
                normalized_dic[sig] = (denormed_dic[sig] - normalizations[sig]['mean']) / normalizations[sig]['std']
        else:
            normalized_dic[sig] = denormed_dic[sig]
    return normalized_dic

def get_denormalized_dic(normed_dic, excluded_sigs=[]):
    denormalized_dic={}
    for sig in normed_dic:
        normed_dic[sig]=np.array(normed_dic[sig])
        if sig not in excluded_sigs:
            if 'qpsi' in sig:
                denormalized_dic[sig] = 1. / normed_dic[sig]
            else:
                denormalized_dic[sig] = (normed_dic[sig] * normalizations[sig]['std']) + normalizations[sig]['mean']
        else:
            denormalized_dic[sig] = normed_dic[sig]
    if use_gyroBohm:
        denormalized_dic=get_gyro_denormalized_dic(denormalized_dic)
    return denormalized_dic
