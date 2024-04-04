import configparser
import torch
import h5py
#torch.manual_seed(0)
import customDatasetMakers
import dataSettings
import numpy as np
import sys
import os
import prediction_helpers
import pickle
from train_helpers import make_bucket
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from dataSettings import get_denormalized_dic,normalizations
from customDatasetMakers import state_to_dic
from scipy import stats
import copy
from aggregate import inference_model, train_model
# for fake actuators
from customDatasetMakers import get_state_indices_dic

import matplotlib.pyplot as plt
import matplotlib

SMALL_SIZE = 25
MEDIUM_SIZE = 30
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines',linewidth=4)

import time

MAX_NUMBER_OF_PREDICTIONS=100000
MAX_NUMBER_OF_TIMES=300

def extract_chains(array, min_length=1):
    chains = []
    chain_indices = []
    current_chain = []
    start_index = None
    for index, value in enumerate(array):
        if np.isnan(value):
            if len(current_chain)>0:  # Check if the current chain is not empty
                if len(current_chain)>=min_length:
                    chains.append(current_chain)
                    chain_indices.append((start_index, index - 1))
                current_chain = []  # Reset the current chain
                start_index = None
        else:
            if start_index is None:
                start_index = index
            current_chain.append(value)
    if current_chain:  # Add the last chain if it exists
        chains.append(current_chain)
        chain_indices.append((start_index, index))
    return chains, chain_indices

def get_ml_truth(x_test,y_test,
                 profiles, parameters,
                 recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho'],
                 prediction_length=20, nwarmup=0, use_fancy_normalization=False):
    num_samples=len(y_test)
    num_profiles=len(profiles)
    # just make this bigger than you think it needs to be
    y=np.ones((num_samples,num_profiles,MAX_NUMBER_OF_TIMES,dataSettings.nx))*np.nan
    for sample_ind in range(num_samples):
        output_dic=state_to_dic(y_test[sample_ind], profiles, parameters)
        #### get input stuff (profile warmup and actuator trajectories
        # only needed for 
        input_dic=state_to_dic(x_test[sample_ind], profiles, parameters, calculations, actuators)
        if use_fancy_normalization:
            for actuator in actuators:
                output_dic[actuator]=input_dic[actuator][:,0]
        ####
        denormed_dic=get_denormalized_dic(output_dic, use_fancy_normalization=use_fancy_normalization)
        for profile_ind,profile in enumerate(recorded_profiles):
            num_times=len(denormed_dic[profile][nwarmup:])
            y[sample_ind,profile_ind,:num_times]=denormed_dic[profile][nwarmup:]
    return y[:,:,:prediction_length]
        #for sig in parameters:
        # remember this returns actuators at the present AND NEXT time, hence -1 index below        
        # input_dic=state_to_dic(x_test[sample_ind], profiles, parameters, calculations, actuators)
        # denormed_dic=get_denormalized_dic(input_dic)
        # for sig in actuators:
        #     all_info[key]['truth']['actuators'][sig]=denormed_dic[sig][:,-1]
        #     all_info[key]['normed_truth']['actuators'][sig]=input_dic[sig][:,-1]
        
def get_ml_profile_warmup_and_actuator_trajectory(x_test,
                                                  profiles, parameters, calculations, actuators,
                                                  recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho'],
                                                  recorded_actuators=['pinj'],
                                                  prediction_length=15, nwarmup=0, use_fancy_normalization=False):
    num_samples=len(x_test)
    profile_warmup=np.ones((num_samples,len(recorded_profiles),nwarmup+1,dataSettings.nx))*np.nan
    # make this bigger than you think is necessary
    actuator_trajectory=np.ones((num_samples,len(recorded_actuators),MAX_NUMBER_OF_TIMES))*np.nan
    for sample_ind in range(num_samples):
        output_dic=state_to_dic(x_test[sample_ind], profiles, parameters, calculations, actuators)
        denormed_dic=get_denormalized_dic(output_dic, use_fancy_normalization=use_fancy_normalization)
        for profile_ind,profile in enumerate(recorded_profiles):
            profile_warmup[sample_ind,profile_ind]=denormed_dic[profile][:nwarmup+1]
        for actuator_ind,actuator in enumerate(recorded_actuators):
            num_times=len(denormed_dic[actuator])
            actuator_trajectory[sample_ind,actuator_ind,:num_times]=denormed_dic[actuator][:,0]
            actuator_trajectory[sample_ind,actuator_ind,num_times]=denormed_dic[actuator][-1,1]
    return profile_warmup, actuator_trajectory[:,:,:prediction_length+nwarmup+1]

def get_ml_predictions(x_test, y_test,
                profiles, parameters, calculations, actuators,
                considered_models,
                recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho'],
                recorded_actuators=['pinj'],
                prediction_length=15,nwarmup=0,
                use_fancy_normalization=False,
                num_rollout_steps=400,
                bucket_size=10000):
    test_x_buckets = make_bucket(x_test, bucket_size)
    test_y_buckets = make_bucket(y_test, bucket_size)
    test_length_buckets = [[len(arr) for arr in bucket] for bucket in test_x_buckets]
    # used to help index stuff later
    running_num_samples=np.insert(np.cumsum([len(bucket) for bucket in test_x_buckets]),0,0)
    num_keys=len(x_test)
    num_profiles=len(recorded_profiles)
    yhat=np.ones((num_keys,num_profiles,MAX_NUMBER_OF_TIMES,dataSettings.nx))*np.nan
    yhat_error=np.ones((num_keys,num_profiles,MAX_NUMBER_OF_TIMES,dataSettings.nx))*np.nan
    begin_time=time.time()
    prev_time=begin_time
    evaluation_begin_time=time.time()
    prev_time=evaluation_begin_time
    with torch.no_grad():
        sample_ind=0
        for which_bucket in range(len(test_x_buckets)):
            x_bucket=test_x_buckets[which_bucket]
            y_bucket=test_y_buckets[which_bucket]
            length_bucket=test_length_buckets[which_bucket]
            padded_x=pad_sequence(x_bucket, batch_first=True)
            padded_y=pad_sequence(y_bucket, batch_first=True)
            #padded_x=padded_x.to(device)
            #padded_y=padded_y.to(device)
            # only save simulations after warmup is over
            # see note above, taking out ability to ensemble models
            # since the ethos should be considering different ML and sim
            # models on equal footing
            model_output=torch.zeros_like(padded_y)
            for model in considered_models:
                #model=considered_models[0]
                model_output+=model(padded_x, reset_probability=0, nwarmup=nwarmup)
            model_output/=len(considered_models)
            unpadded_output=unpad_sequence(model_output, length_bucket, batch_first=True)
            for which_output,output in enumerate(unpadded_output):
                output_dic=state_to_dic(output, profiles, parameters)
                #### get input stuff (profile warmup and actuator trajectories
                # only needed for 
                input_dic=state_to_dic(x_test[sample_ind], profiles, parameters, calculations, actuators)
                if use_fancy_normalization:
                    for actuator in actuators:
                        output_dic[actuator]=input_dic[actuator][:,-1]
                ####
                denormed_dic=get_denormalized_dic(output_dic, use_fancy_normalization=use_fancy_normalization)
                for profile_ind,profile in enumerate(recorded_profiles):
                    num_times=len(denormed_dic[profile][nwarmup:])
                    yhat[sample_ind,profile_ind,:num_times]=denormed_dic[profile][nwarmup:]
                sample_ind+=1
                #for sig in parameters:
                    #yhat[sample_ind,profile_ind,:prediction_length]=denormed_dic[sig][nwarmup:prediction_length+nwarmup]
            print(f'Bucket {which_bucket+1}/{len(test_x_buckets)} took {time.time()-prev_time:0.0f}s')
            prev_time=time.time()
    print(f'Took {time.time()-begin_time:.2f} s')
    return yhat[:,:,:prediction_length]

# use_delta and return_truth are obsolete with newer versions, where I deal with that stuff on the ASTRA side
# max_num_shots is helpful for testing
def get_sim_predictions_shots_times(sim_name, sim_dir, prediction_length,
                                    recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho'],
                                    min_length=5,
                                    ntimestep_delay=0,
                                    use_delta=False,
                                    max_num_shots=None,
                                    return_truth=False):
    h5_path=os.path.join(sim_dir,sim_name+'.h5')
    name_map={'zipfit_etempfit_rho': 'TE', 'zipfit_itempfit_rho': 'TI', 'zipfit_trotfit_rho': 'UPAR', 'zipfit_edensfit_rho': 'NE', 'qpsi_EFIT01': 'MU',
              'zeff_rho': 'ZEF'}
    recorded_profile_astra_names=[name_map[profile] for profile in recorded_profiles]
    experiment_names={'TE': 'TEX', 'TI': 'TIX', 'UPAR': 'VTORX', 'NE': 'NEX', 'MU': 'MUX', 'ZEF': 'ZEF'}
    y=np.ones((MAX_NUMBER_OF_PREDICTIONS,len(recorded_profiles),MAX_NUMBER_OF_TIMES,dataSettings.nx))*np.nan
    yhat=np.ones((MAX_NUMBER_OF_PREDICTIONS,len(recorded_profiles),MAX_NUMBER_OF_TIMES,dataSettings.nx))*np.nan
    with h5py.File(h5_path) as f:
        print('loading h5')
        shots=list(f.keys())
        if max_num_shots is not None:
            shots=shots[:max_num_shots]
        print('h5 loaded, reading in simulation data')
        key_ind=0
        sim_times=[]
        sim_shots=[]
        trajectory_lengths=[]
        for nshot,shot in enumerate(shots):
            # in future might want to make this more lenient, for now force to have the right number of timesteps
            _,indices=extract_chains(f[shot][f'TE_{sim_name}'][:,0],min_length=min_length)
            for indices_index in range(len(indices)):
                # start_index is the time from which the first prediction is made
                # we also save 1 point before this hence +1 throughout
                start_index=indices[indices_index][0]+ntimestep_delay
                end_index=indices[indices_index][1]+1
                trajectory_lengths.append(end_index-start_index)
                start_time=int(start_index*dataSettings.DT*1e3)
                sim_times.append(start_time)
                sim_shots.append(int(shot))
                num_available_prediction_times=end_index-(start_index+1)
                for profile_ind, profile in enumerate(recorded_profile_astra_names):
                    expt_profile=experiment_names[profile]
                    if use_delta:
                        # usually this is used for ntimestep_delay
                        yhat[key_ind,profile_ind,:num_available_prediction_times]=f[shot][f'{expt_profile}_{sim_name}'][start_index]+\
                            (f[shot][f'{profile}_{sim_name}'][start_index+1:end_index]-f[shot][f'{profile}_{sim_name}'][start_index])
                    else:
                        yhat[key_ind,profile_ind,:num_available_prediction_times]=f[shot][f'{profile}_{sim_name}'][start_index+1:end_index]
                    if return_truth:
                        y[key_ind,profile_ind,:num_available_prediction_times]=f[shot][f'{expt_profile}_{sim_name}'][start_index+1:end_index]
                    #all_info[key]['truth']['profiles'][name_map[predicted_sig]]=f[shot][f'{experiment_names[predicted_sig]}_{sim_name}'][start_index:end_index]
                    if profile=='MU':
                        yhat[key_ind,profile_ind]=1./yhat[key_ind,profile_ind]
                        if return_truth:
                            y[key_ind,profile_ind]=1./yhat[key_ind,profile_ind]
                    if profile=='UPAR':
                        upar_scaling=1./(1.e3*f[shot][f'rgeo_{sim_name}'][start_index+1:end_index][:,None])
                        yhat[key_ind,profile_ind,:len(upar_scaling)]=yhat[key_ind,profile_ind,:len(upar_scaling)]*upar_scaling
                        if return_truth:
                            y[key_ind,profile_ind,:len(upar_scaling)]=y[key_ind,profile_ind,:len(upar_scaling)]*upar_scaling
                key_ind+=1
        unique, counts = np.unique(trajectory_lengths, return_counts=True)
        print(dict(zip(unique,counts)))
    print(f'Read in {len(sim_shots)} simulation rollouts')
    return yhat[:key_ind,:,:prediction_length], sim_shots, sim_times, y[:key_ind,:,:prediction_length]

# takes info of form {dataset: {shots: [...], times: [...], data: [...]}} where ... is over samples
# updates all 3 arrays of each dataset (in place) to have shared shot_times across datasets and be sorted
def subsample_info_to_shared_keys(all_info):
    print('Subsampling data to match shots/times, lengths of each dataset are:')
    print({dataset: len(all_info[dataset]['data']) for dataset in all_info})
    shot_time_keys={dataset: [f"{all_info[dataset]['shots'][ind]}_{all_info[dataset]['times'][ind]}"
                             for ind in range(len(all_info[dataset]['shots']))]
                    for dataset in all_info}
    shared_keys=sorted(list(set.intersection(*[set(shot_time_keys[dataset]) for dataset in all_info])))
    shots=[int(key.split('_')[0]) for key in shared_keys]
    times=[int(key.split('_')[1]) for key in shared_keys]
    new_indices={dataset: [shot_time_keys[dataset].index(key) for key in shared_keys] for dataset in shot_time_keys}
    for dataset in all_info:
        all_info[dataset]['shots']=shots
        all_info[dataset]['times']=times
        all_info[dataset]['data']=all_info[dataset]['data'][new_indices[dataset]]
    num_samples=len(shots)
    print(f'{num_samples} samples from {len(np.unique(shots))} unique shots shared between {all_info.keys()}')
    return shots,times

if __name__ == "__main__":
    raw_data_filename='/projects/EKOLEMEN/profile_predictor/raw_data/diiid_data.h5' #small_test.h5'
    use_ensemble=False
    profiles=['zipfit_etempfit_rho', 'zipfit_itempfit_rho', 'zipfit_trotfit_rho',
              'zipfit_edensfit_rho', 'zipfit_zdensfit_rho', 'qpsi_EFIT01']
    ip_minimum=1.0e6
    ip_maximum=1.2e6
    nwarmup=3
    recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho',
                       'zipfit_edensfit_rho', 'qpsi_EFIT01','zeff_rho']
    recorded_actuators=['pinj','ip','volume_EFIT01','rmaxis_EFIT01','aminor_EFIT01']
    prediction_length=15
    include_const_predictions=False
    plotted_profiles=recorded_profiles
    plotted_actuators=recorded_actuators
    plot_sigma_bar=False
    plot_sigma_time=False
    plot_over_rho=False
    plot_over_time=False
    model_blends={}
    model_colors={}
    model_name_map={}
    model_linestyles={'const': '--'} #, 'ip_0_1200NOdssdenest_RESUMEDconfig': '--'}
    model_name_map={'const': 'constant',
                    #'ip_0_1200NOdssdenest_RESUMED3config': 'ML (ip<1200kA)',
                    'ip_0_900NOdssdenest_RESUMED3config': r'ML ($I_p$<0.9MA)',
                    'allNOdssdenest_RESUMED3config': 'ML (all)',
                    'ip_0_1200WITHdssdenest_RESUMED3config': r'ML ($I_p$<1.2MA + <$n_e$>)', 'ip_0_900WITHdssdenest_RESUMED3config': r'ML ($I_p$<0.9MA + <$n_e$>)',
                    # surrogate hybrid stuff
                    'ip_0_1200NOdssdenest_RESUMED3config': r'ML ($I_p$<1.2MA)', #'untuned',
                    'surrogateHybrid_tuned_on_data_only_ip_0_900unfrozenconfig': 'data\n(all)', 'surrogateHybridip_0_900unfrozenconfig': 'transfer\n(all)',
                    'surrogateHybrid_tuned_on_data_only_ip_0_900frozenEncodersconfig': 'data\n(RNN)', 'surrogateHybridip_0_900frozenEncodersconfig': 'transfer\n(RNN)', 
                    'surrogateHybrid_tuned_on_data_only_ip_0_900frozenRNNconfig': 'data\n(Enc/Dec)', 'surrogateHybridip_0_900frozenRNNconfig': 'transfer\n(Enc/Dec)',
                    # curriculum stuff
                    'alldiiid_ensembleconfig0EPOCH250': r'$\mu$=20ms prediction','alldiiid_ensembleconfig0EPOCH500': r'Epoch 500: $\mu$=100ms prediction',
                    'alldiiid_ensembleconfig0EPOCH750': r'Epoch 750: $\mu$=200ms prediction','alldiiid_ensembleconfig0': r'$\mu$=200ms prediction',
                    # aug
                    'augall_d3d900NOdssdenestconfig': r'$I_p$<0.9MA+AUG',
                    'augall_d3d900NOdssdenestNORMEDconfig': r'$I_p$<0.9MA'+'\n+AUG normed', 'augall_d3d900NOdssdenestUNNORMEDconfig': r'$I_p$<0.9MA'+'\n+AUG',
                    'augallNOdssdenestwithGBnormalizationconfig': 'AUG normed', 'augallNOdssdenestnoGBnormalizationconfig': 'AUG',
                    # calcs
                    'astraInterpretiveAndTGLFNNallnoCalcsconfig': 'data only',
                    'astraInterpretiveAndTGLFNNallwithPredictiveconfig': '+tglfnn',
                    'astraInterpretiveAndTGLFNNallwithInterpretiveconfig': '+interpreted',
                    # sims
                    'astrapredictFIXEDZIPFIT': 'fixed', 'astrapredictFIXEDGBZIPFIT': 'fixed+GB', 'astrapredictFIXEDTGLFNNZIPFIT': 'fixed+tglfnn',
                    'astrapredictTGLFNNZIPFIT': 'tglfnn',
                    'astrapredictTGLFNNEPEDNNZIPFIT': 'tglfnn+epednn',
                    'astrapredictTGLFNNandScaleDensityZIPFIT': 'tglfnn-FIXED+scale-NE',
                    'astrapredictFULLYZIPFIT': 'tglfnn+eped+NE',
                    'astrapredictFULLYandCurrentZIPFIT': 'tglfnn+eped+NE+q',
                    'astrapredictTGLFNNandCurrentZIPFIT': 'tglfnn+q',
                    'astrapredictTGLFNNZIPFIT': 'tglfnn'}
    sigma_bar_title=r'$\sigma$ error on $1.0MA<I_p<1.2MA$'

    common_blend_info={}
    #common_blend_info['BlenderNonlinear']={'model_type': 'BlenderNonlinear', 'model_filename': 'blenderNonlinear.tar'}
    # common_blend_info['SimpleAverage']={'model_type': 'SimpleAverage'}
    common_blend_info['Blender']={'model_type': 'Blender', 'model_filename': 'blender.tar'}
    # common_blend_info['BlenderProfiles']={'model_type': 'BlenderProfiles', 'model_filename': 'blenderProfiles.tar'}
    # common_blend_info['BlenderProfilesTimes']={'model_type': 'BlenderProfilesTimes', 'model_filename': 'blenderProfilesTimes.tar'}
    for blend in common_blend_info:
        common_blend_info[blend]['relevant_models']=['astrapredictTGLFNNZIPFIT','astrapredictFIXEDTGLFNNZIPFIT','astrapredictFIXEDGBZIPFIT']
        common_blend_info[blend]['relevant_profiles']=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho']
    model_colors['Blender']='m'
    model_name_map['Blender']='meta'
    # comparing d3d to aug to gyrobohm normalized
    if False:
        plotted_profiles=['zipfit_itempfit_rho','zipfit_trotfit_rho','zipfit_edensfit_rho'] #'zipfit_etempfit_rho',
        sigma_bar_title='Error (%)'
        plot_sigma_bar=True
        considered_sims=[]
        #'augip_0_1200NOdssdenestconfig', 'aug900_d3d900NOdssdenestconfig',
        ml_configs=[#'ip_0_1200NOdssdenest_RESUMED3config',
            'ip_0_900NOdssdenest_RESUMED3config',
            'augall_d3d900NOdssdenestUNNORMEDconfig',
            'augall_d3d900NOdssdenestNORMEDconfig']
                    #'augallNOdssdenestnoGBnormalizationconfig',
                    #'augallNOdssdenestwithGBnormalizationconfig']
        model_name_map.update({'ip_0_900NOdssdenest_RESUMED3config': 'D3D',#r'$I_p$<0.9MA',
                               #'augall_d3d900NOdssdenestconfig': r'D3D+AUG',
                               'augall_d3d900NOdssdenestNORMEDconfig': 'Normed\nD3D+AUG',
                               'augall_d3d900NOdssdenestUNNORMEDconfig': r'D3D+AUG', #r'$I_p$<0.9MA'+'\n+AUG',
                               'augallNOdssdenestwithGBnormalizationconfig': 'AUG normed',
                               'augallNOdssdenestnoGBnormalizationconfig': 'AUG',
                               'ip_0_1200NOdssdenest_RESUMED3config': r'$I_p$<1.2MA'})
        ml_cache_filename='/scratch/gpfs/jabbate/ml_aug_comparison.pkl'
        data_cache_filename='/scratch/gpfs/jabbate/data_1000_1200.pkl'
    # surrogate hybrid (tuning on simulation outputs)
    elif False:
        sigma_bar_title='Error (%)'
        plot_sigma_bar=True
        model_name_map.update({'ip_0_900NOdssdenest_RESUMED3config': 'ML', #r'$I_p$<0.9MA',
                               'ip_0_1200NOdssdenest_RESUMED3config': r'$I_p$<1.2MA'})
        plotted_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho']
        considered_sims=[]
        #'augip_0_1200NOdssdenestconfig', 'aug900_d3d900NOdssdenestconfig',
        ml_configs=['ip_0_900NOdssdenest_RESUMED3config',
                    #'surrogateHybrid_tuned_on_data_only_ip_0_900frozenEncodersconfig',
                    'surrogateHybridip_0_900frozenEncodersconfig',
                    #'surrogateHybrid_tuned_on_data_only_ip_0_900frozenRNNconfig',
                    'surrogateHybridip_0_900frozenRNNconfig',
                    #'surrogateHybrid_tuned_on_data_only_ip_0_900unfrozenconfig',
                    'surrogateHybridip_0_900unfrozenconfig']
        ml_cache_filename='/scratch/gpfs/jabbate/ml_surrogate_hybrid.pkl'
        data_cache_filename='/scratch/gpfs/jabbate/data_1000_1200.pkl'
    # comparing sims for 1.0 to 1.2, extracting file to train coefficients on
    elif False:
        plot_sigma_bar=True
        plotted_profiles=recorded_profiles[:3]
        plotted_actuators=recorded_actuators[:1]
        #considered_sims=[]
        considered_sims=['astrapredictTGLFNNZIPFIT', #'astrapredictTGLFNNEPEDNNZIPFIT',
                         'astrapredictFIXEDGBZIPFIT','astrapredictFIXEDTGLFNNZIPFIT']
                         #'astrapredictTGLFNNandScaleDensityZIPFIT']
                         #'astrapredictFULLYZIPFIT',
                         #'astrapredictTGLFNNandScaleDensityZIPFIT', 'astrapredictFULLYandCurrentZIPFIT']
        ml_configs=['ip_0_1200NOdssdenest_RESUMED3config','ip_0_900NOdssdenest_RESUMED3config']
        train_blends=copy.deepcopy(common_blend_info)
        for blend in train_blends:
            train_blends[blend]['retrain']=True
            train_blends[blend]['relevant_models'].append('ip_0_900NOdssdenest_RESUMED3config')
        model_blends.update(train_blends)
        ip_minimum=1.0e6
        ip_maximum=1.2e6
        data_cache_filename='/scratch/gpfs/jabbate/data_sim_1000_1200.pkl' #+'_'.join(considered_sims)+'.pkl'
        ml_cache_filename='/scratch/gpfs/jabbate/ml_sim_1000_1200.pkl' #+'_'.join(considered_sims)+'.pkl'
        raw_data_filename='/projects/EKOLEMEN/profile_predictor/raw_data/diiid_data.h5' #small_test.h5'
        model_colors['ensemble\n(average)']='r'
        model_colors['Blender']='m'
    # comparing for 1.3 and up, using trained coefficients
    elif False:
        model_name_map.update({'ip_0_1200NOdssdenest_RESUMED3config': 'ML'})
        sigma_bar_title='Error (%)' #r'$\sigma$ error on $1.3MA<I_p$'
        plotted_profiles=recorded_profiles[:3]
        plotted_actuators=[] #recorded_actuators[:1]
        #considered_sims=[]
        considered_sims=['astrapredictTGLFNNZIPFIT', #'astrapredictTGLFNNEPEDNNZIPFIT',
                         'astrapredictFIXEDGBZIPFIT','astrapredictFIXEDTGLFNNZIPFIT']
                         #'astrapredictTGLFNNandScaleDensityZIPFIT']
                         #'astrapredictFULLYZIPFIT',
                         #'astrapredictTGLFNNandScaleDensityZIPFIT', 'astrapredictFULLYandCurrentZIPFIT']
        # for individual example cases
        test_blends=copy.deepcopy(common_blend_info)
        for blend in test_blends:
            print('Make sure you already ran this code for the case that trains the weights in the blend')
            test_blends[blend]['retrain']=False
            test_blends[blend]['relevant_models'].append('ip_0_1200NOdssdenest_RESUMED3config')
        model_blends.update(test_blends)
        if True:
            plotted_profiles=['zipfit_etempfit_rho','zipfit_trotfit_rho']
            model_name_map.update({'ip_0_900NOdssdenest_RESUMED3config': 'ML',
                                   'ip_0_1200NOdssdenest_RESUMED3config': 'ML'})
            ml_configs=['ip_0_1200NOdssdenest_RESUMED3config']
            plot_over_time=True
            plot_over_rho=True
            plot_sigma_bar=False
        # for sigma error comparison
        else:
            plot_over_time=False
            plot_over_rho=False
            plot_sigma_bar=True
            ml_configs=['ip_0_1200NOdssdenest_RESUMED3config'] #['allNOdssdenest_RESUMED3config','ip_0_1200NOdssdenest_RESUMED3config']
        ip_minimum=1.3e6
        ip_maximum=10e6
        data_cache_filename='/scratch/gpfs/jabbate/data_sim_1300.pkl' #+'_'.join(considered_sims)+'.pkl'
        ml_cache_filename='/scratch/gpfs/jabbate/ml_sim_1300.pkl' #+'_'.join(considered_sims)+'.pkl'
        raw_data_filename='/projects/EKOLEMEN/profile_predictor/raw_data/diiid_data.h5' #small_test.h5'
        # Take this from running the block above this to dump the thing to train on, then aggregate.py
        model_colors['ensemble\n(average)']='tab:pink'
        model_colors['ensemble\n(optimized)']='m'
    # seeing whether adding ASTRA calculations helps
    elif False:
        plotted_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','qpsi_EFIT01']
        sigma_bar_title='Error (%)' #r'$\sigma$ error on all data'
        plot_sigma_bar=True
        legend_fontsize=10
        ml_configs=['astraInterpretiveAndTGLFNNallnoCalcsconfig','astraInterpretiveAndTGLFNNallwithPredictiveconfig',
                    'astraInterpretiveAndTGLFNNallwithInterpretiveconfig']
        considered_sims=[] #['astrapredictFIXEDGBZIPFIT', 'astrapredictFIXEDTGLFNNZIPFIT','astrapredictTGLFNNlowipZIPFIT']
        data_cache_filename='/scratch/gpfs/jabbate/data_calculations.pkl'
        ip_minimum=None
        ip_maximum=None
        raw_data_filename='/projects/EKOLEMEN/profile_predictor/sim_data/astraTrainData.h5'
        profiles=['zipfit_etempfit_rho', 'zipfit_itempfit_rho', 'zipfit_trotfit_rho',
                  'zipfit_edensfit_rho', 'zipfit_zdensfit_rho', 'qpsi_EFIT01',
                  'PETOT_astrainterpretZIPFIT','PITOT_astrainterpretZIPFIT','CD_astrainterpretZIPFIT',
                  'TE_astrapredictTGLFNNZIPFIT','TI_astrapredictTGLFNNZIPFIT']
        ml_cache_filename='/scratch/gpfs/jabbate/ml_calcs_comparison.pkl'
    # ensemble stuff for explaining ensembling
    elif False:
        considered_sims=[]
        plotted_profiles=recorded_profiles
        # JANK: pop a "EPOCH***" on the end to use a specific epoch of a config file
        ml_configs=['alldiiid_ensembleconfig0','alldiiid_ensembleconfig1','alldiiid_ensembleconfig2',
                    'alldiiid_ensembleconfig0EPOCH250','alldiiid_ensembleconfig1EPOCH250','alldiiid_ensembleconfig2EPOCH250']
        ml_cache_filename='/scratch/gpfs/jabbate/ml_ensemble.pkl'
        data_cache_filename='/scratch/gpfs/jabbate/data_1000_1200.pkl'
        model_blends={'200ms': {'coefficients': [0.5, 0.5], 'models': ['alldiiid_ensembleconfig0','alldiiid_ensembleconfig1']},
                      '20ms': {'coefficients': [0.5, 0.5], 'models': ['alldiiid_ensembleconfig0EPOCH250','alldiiid_ensembleconfig1EPOCH250']}}
        model_colors.update({'alldiiid_ensembleconfig0': 'r','alldiiid_ensembleconfig1': 'r','alldiiid_ensembleconfig2':'r',
                             'alldiiid_ensembleconfig0EPOCH250':'b','alldiiid_ensembleconfig1EPOCH250':'b','alldiiid_ensembleconfig2EPOCH250':'b',
                             '200ms': 'r', '20ms': 'b'})
        prediction_length=25
        plotted_actuators=recorded_actuators[:1]
    # ensemble stuff for explaining curriculum learning
    elif True:
        #plotted_profiles=recorded_profiles
        plotted_profiles=['zipfit_etempfit_rho']
        plot_sigma_bar=False
        plot_sigma_time=True
        plot_over_rho=False
        plot_over_time=False
        considered_sims=[]
        # JANK: pop a "EPOCH***" on the end to use a specific epoch of a config file
        ml_configs=['alldiiid_ensembleconfig0EPOCH250',#'alldiiid_ensembleconfig0EPOCH500',
                    #'alldiiid_ensembleconfig0EPOCH750',
                    'alldiiid_ensembleconfig0']
        use_ensemble=False
        ml_cache_filename='/scratch/gpfs/jabbate/ml_curriculum.pkl'
        data_cache_filename='/scratch/gpfs/jabbate/data_1000_1200.pkl'
        # model_blends={'200ms': {'coefficients': [0.5, 0.5], 'models': ['alldiiid_ensembleconfig0','alldiiid_ensembleconfig1']},
        #               '20ms': {'coefficients': [0.5, 0.5], 'models': ['alldiiid_ensembleconfig0EPOCH250','alldiiid_ensembleconfig1EPOCH250']}}
        # model_colors.update({'alldiiid_ensembleconfig0': 'r','alldiiid_ensembleconfig1': 'r','alldiiid_ensembleconfig2':'r',
        #                      'alldiiid_ensembleconfig0EPOCH250':'b','alldiiid_ensembleconfig1EPOCH250':'b','alldiiid_ensembleconfig2EPOCH250':'b',
        #                      '200ms': 'r', '20ms': 'b'})
        model_colors.update({'alldiiid_ensembleconfig0EPOCH250': 'r','alldiiid_ensembleconfig0EPOCH500': 'b',
                             'alldiiid_ensembleconfig0EPOCH750': 'b','alldiiid_ensembleconfig0': 'b'})
        prediction_length=15
        plotted_actuators=recorded_actuators[:1]
        #model_linestyles.update({'alldiiid_ensembleconfig0': '--'})
    sim_color_map=matplotlib.colormaps['winter'](np.linspace(0,1,len(considered_sims)))
    ml_color_map=matplotlib.colormaps['autumn'](np.linspace(0,1,len(ml_configs)))
    for i,model_name in enumerate(considered_sims):
        if model_name not in model_colors:
            model_colors[model_name]=sim_color_map[i]
    for i,model_name in enumerate(ml_configs):
        if model_name not in model_colors:
            model_colors[model_name]=ml_color_map[i]
    #model_colors['alldiiid_ensembleconfigEPOCH250']='r'
    #model_colors['alldiiid_ensembleconfigEPOCH500']='b'
    sig_name_map={'zipfit_etempfit_rho': r'$T_e$', 'zipfit_itempfit_rho': r'$T_i$', 'zipfit_trotfit_rho': r'$\Omega$',
                  'zipfit_edensfit_rho': r'$n_e$', 'zeff_rho': r'$Z_{eff}$', 'qpsi_EFIT01': '$q$',
                  'pinj': r'$P_{inj}$', 'ip': r'$I_p$'}
    sim_dir="/projects/EKOLEMEN/profile_predictor/sim_data/"
    all_sim_info={}
    for sim_name in considered_sims:
        sim_cache_filename=f'tmp_{sim_name}.pkl'
        if not os.path.exists(sim_cache_filename):
            print(f'making {sim_name} dataset, caching in {sim_cache_filename}')
            if sim_name in ['astrapredictTGLFNNEPEDNNZIPFIT','astrapredictFULLYtglfnnZIPFIT']:
                ntimestep_delay=5
                use_delta=True
            else:
                ntimestep_delay=0
                use_delta=False
            sim_yhat, sim_shots, sim_times, sim_y=get_sim_predictions_shots_times(sim_name, sim_dir, prediction_length=prediction_length,
                                                                                  recorded_profiles=recorded_profiles,
                                                                                  ntimestep_delay=ntimestep_delay,
                                                                                  min_length=15,
                                                                                  use_delta=use_delta)
            sim_info={'shots': sim_shots, 'times': sim_times, 'data': sim_yhat}
            sim_truth_info={'shots': sim_shots, 'times': sim_times, 'data': sim_y}
            all_sim_info[sim_name]=sim_info
            with open(sim_cache_filename,'wb') as f:
                pickle.dump(sim_info,f)
        else:
            print(f'drawing {sim_name} from {sim_cache_filename}, delete to remake')
            with open(sim_cache_filename,'rb') as f:
                all_sim_info[sim_name]=pickle.load(f)
    if True:
        if len(considered_sims)>0:
            print('Computing dataset with simulation shots/timebounds')
            shots_to_preprocess,sim_times=subsample_info_to_shared_keys(all_sim_info)
            time_bounds_to_preprocess=[]
            # ML's first output is 20ms ahead of the start time
            # similarly if we want the last prediction we have to get one extra
            for sample_ind in range(len(sim_times)):
                time_bounds_to_preprocess.append([sim_times[sample_ind]-nwarmup*dataSettings.DT*1.e3,
                                                  sim_times[sample_ind]+prediction_length*dataSettings.DT*1.e3])
        else:
            min_shot=140000
            max_shot=200000
            test_index=0
            shots_to_preprocess=[shot for shot in range(min_shot,max_shot) if shot%10 in [test_index]]
            time_bounds_to_preprocess=None
        scalars=['pinj','tinj','ech_pwr_total','ip','tribot_EFIT01','tritop_EFIT01','kappa_EFIT01','aminor_EFIT01',
                 'rmaxis_EFIT01','volume_EFIT01','bt','D_tot','H_tot','He_tot','N_tot','Ne_tot',
                 'dssdenest']
        if os.path.exists(data_cache_filename):
            print(f'{data_cache_filename} already written, delete to remake')
        else:
            customDatasetMakers.preprocess_data(data_cache_filename,
                                                raw_data_filename,profiles,scalars,
                                                shots=shots_to_preprocess, time_bounds=time_bounds_to_preprocess,
                                                exclude_ech=False,
                                                ip_minimum=ip_minimum,ip_maximum=ip_maximum,
                                                zero_fill_signals=['ech_pwr_total','pinj','tinj'])
    # now get the models and dump the predictions
    if not os.path.exists(ml_cache_filename):
        ml_model_dirname='/projects/EKOLEMEN/profile_predictor/final_paper_models/'
        all_ml_info={}
        for ml_config in ml_configs:
            # super jank: name it like basenameconfigEPOCH500
            config_name_info=ml_config.split('EPOCH')
            if len(config_name_info)>1:
                epoch=int(config_name_info[1])
                base_ml_config=config_name_info[0]
            else:
                epoch=None
                base_ml_config=ml_config
            ensemble=use_ensemble
            config_filename=os.path.join(ml_model_dirname, base_ml_config)
            config=configparser.ConfigParser()
            config.read(config_filename)
            profiles=config['inputs']['profiles'].split()
            actuators=config['inputs']['actuators'].split()
            parameters=config['inputs'].get('parameters','').split()
            calculations=config['inputs'].get('calculations','').split()
            use_fancy_normalization=config['preprocess'].getboolean('use_fancy_normalization',False)
            fake_actuators=False
            num_rollout_steps=400
            min_sample_length=nwarmup+1 #num_rollout_steps+nwarmup
            x_test, y_test, ml_shots, times =customDatasetMakers.ian_dataset(data_cache_filename,profiles,parameters,calculations,actuators,sort_by_size=True,
                                                                             min_sample_length=min_sample_length,
                                                                             use_fancy_normalization=use_fancy_normalization)
            if False:
                state_indices=get_state_indices_dic(profiles,parameters,calculations=calculations,actuators=actuators)
                for i in range(len(x_test)):
                    for actuator in actuators:
                        index_0=state_indices[actuator][0]
                        index_1=state_indices[actuator][1]
                        x_test[i][:,index_0]=x_test[i][nwarmup,index_0]
                        x_test[i][:,index_1]=x_test[i][nwarmup,index_0]
                    for profile in profiles:
                        indices=state_indices[profile]
                        x_test[i][:nwarmup,indices]=x_test[i][nwarmup,indices]
                        x_test[i][:nwarmup,indices]=x_test[i][nwarmup,indices]                        
            ml_times=np.array(times)+nwarmup*dataSettings.DT*1.e3
            ml_times=ml_times.astype(int)
            start_times=ml_times
            # ml prediction stuff
            considered_models=prediction_helpers.get_considered_models(config_filename, ensemble=ensemble, epoch=epoch)
            considered_models=considered_models
            ml_predictions=get_ml_predictions(x_test,y_test,
                                              profiles, parameters, calculations, actuators,
                                              considered_models,
                                              recorded_profiles=recorded_profiles,
                                              prediction_length=prediction_length,
                                              nwarmup=nwarmup, use_fancy_normalization=use_fancy_normalization,
                                              num_rollout_steps=num_rollout_steps)
            all_ml_info[ml_config]={'data': ml_predictions, 'shots': ml_shots, 'times': ml_times}
        # #### HOPEFULLY SECOND ML MODEL JUST RUNS
        # config_filename='/projects/EKOLEMEN/profile_predictor/final_paper_models/ip_0_1200WITHdssdenest_RESUMEDconfig'
        # considered_models=prediction_helpers.get_considered_models(config_filename, ensemble=ensemble, epoch=epoch)
        # ml_2_predictions=get_ml_predictions(x_test,
        #                                     profiles, parameters, calculations, actuators,
        #                                     considered_models,
        #                                     nwarmup=nwarmup,
        #                                     num_rollout_steps=num_rollout_steps)
        ####
        # truth stuff -- right now it's jank just uses the stuff from the last model in the ml_configs list
        truth=get_ml_truth(x_test,y_test,
                           profiles, parameters,
                           recorded_profiles=recorded_profiles,
                           prediction_length=prediction_length,
                           nwarmup=nwarmup, use_fancy_normalization=use_fancy_normalization)
        profile_warmup,actuator_trajectory=get_ml_profile_warmup_and_actuator_trajectory(x_test,
                                                                                         profiles, parameters, calculations, actuators,
                                                                                         recorded_profiles=recorded_profiles, recorded_actuators=recorded_actuators,
                                                                                         prediction_length=prediction_length,
                                                                                         nwarmup=nwarmup, use_fancy_normalization=use_fancy_normalization)
        with open(ml_cache_filename,'wb') as f:
            pickle.dump({'all_ml_info': all_ml_info, 'truth': truth, 'profile_warmup': profile_warmup, 'actuator_trajectory': actuator_trajectory,
                         'ml_shots': ml_shots, 'ml_times': ml_times},f)
    else:
        with open(ml_cache_filename,'rb') as f:
            ml_info=pickle.load(f)
            all_ml_info=ml_info['all_ml_info']
            truth=ml_info['truth']
            profile_warmup=ml_info['profile_warmup']
            actuator_trajectory=ml_info['actuator_trajectory']
            ml_shots=ml_info['ml_shots']
            ml_times=ml_info['ml_times']
    all_info={}
    all_info.update(all_sim_info)
    all_info.update(all_ml_info)
    all_info.update({'truth': {'shots': ml_shots, 'times': ml_times, 'data': truth},
                     'profile_warmup': {'shots': ml_shots, 'times': ml_times, 'data': profile_warmup},
                     'actuator_trajectory': {'shots': ml_shots, 'times': ml_times, 'data': actuator_trajectory}})
    shots,times=subsample_info_to_shared_keys(all_info)
    num_samples=len(shots)
    extra_predictions=[]
    extra_prediction_names=[]
    tmp_model_blend_info={}
    for blend in model_blends:
        ### NEW
        # make mask
        # make x
        # make extra_x
        # make y
        # normalize
        relevant_profiles=model_blends[blend]['relevant_profiles']
        model_type=model_blends[blend]['model_type']
        #relevant_extra_info=['']
        profile_inds=[profiles.index(profile) for profile in relevant_profiles]
        ensemble_sims=np.array([all_info[model]['data'][:,profile_inds,:,:] for model in model_blends[blend]['relevant_models']])
        truth=all_info['truth']['data'][:,profile_inds,:,:]
        #extra_info=np.array([all_info['truth']['data'][profile][:,profile_inds,:,:] for profile in relevant_profiles])
        # normalize for the sake of training
        for profile in relevant_profiles:
            profile_ind=relevant_profiles.index(profile)
            ensemble_sims[:,:,profile_ind,:,:]/=normalizations[profile]['std']
            truth[:,profile_ind,:,:]/=normalizations[profile]['std']
        if model_blends[blend]['retrain'] and model_type!='SimpleAverage':
            # train and save it
            ensemble_model=train_model(ensemble_sims,truth,
                                       profiles,relevant_profiles,
                                       model_blends[blend]['model_filename'],
                                       model_blends[blend]['model_type'])
        if model_type=='SimpleAverage':
            yhat=np.mean(ensemble_sims,axis=0)
        else:
            yhat=inference_model(model_blends[blend]['model_filename'],ensemble_sims).detach().numpy()
        blended_predictions=np.zeros_like(all_info['truth']['data'])
        for i,profile in enumerate(relevant_profiles):
            profile_ind=relevant_profiles.index(profile)
            blended_predictions[:,profile_ind,:,:]=yhat[:,i,:,:]*normalizations[profile]['std']
        extra_predictions+=[blended_predictions]
        extra_prediction_names+=[blend]
        # for profile in relevant_profiles:
        #     profile_ind=relevant_profiles.index(profile)
        #     ensemble_sims[:,:,profile_ind,:,:]/=normalizations[profile]['std']
        #     truth[:,profile_ind,:,:]/=normalizations[profile]['std']
        ###
    # if False: #for blend in model_blends:
    #     coefficients=np.array(model_blends[blend]['coefficients'])
    #     relevant_model_info=np.array([all_info[model]['data'] for model in model_blends[blend]['models']])
    #     relevant_model_names=model_blends[blend]['models']
    #     tmp_model_blend_info[blend]={'names': relevant_model_names, 'data': relevant_model_info}
    #     blended_predictions=np.sum(coefficients[:,np.newaxis,np.newaxis,np.newaxis,np.newaxis]*relevant_model_info,axis=0)
    #     extra_predictions+=[blended_predictions]
    #     extra_prediction_names+=[blend]
    if include_const_predictions:
        const_predictions=np.ones((num_samples,len(recorded_profiles),prediction_length,dataSettings.nx))
        for time_ind in range(const_predictions.shape[-2]):
            const_predictions[:,:,time_ind,:]=all_info['profile_warmup']['data'][:,:,-1,:]
        extra_predictions+=[const_predictions]
        extra_prediction_names+=['const']
    tmp_model_blend_info['truth']=all_info['truth']['data']
    tmp_model_blend_info['profiles']=profiles
    with open('tmp_blend_info.pkl','wb') as f:
        pickle.dump(tmp_model_blend_info,f)
    # if include_blended_predictions:
    #     models_for_blend=['ip_0_900NOdssdenest_RESUMED3config', *considered_sims]
    #     blended_predictions=np.sum([all_info[model]['data'] for model in models_for_blend],axis=0)/len(models_for_blend)
    #     extra_predictions+=[blended_predictions]
    #     extra_prediction_names+=['blended']
    ##### edit these 2 lines to change what gets plotted
    model_names=[*ml_configs,*considered_sims,*extra_prediction_names]
    model_predictions=np.stack([*[all_info[model]['data'] for model in ml_configs],
                                *[all_info[model]['data'] for model in considered_sims],
                                *extra_predictions])
    #####
    min_prediction_steps=np.zeros(num_samples).astype(int)
    for sample_ind in range(num_samples):
        for time_ind in reversed(range(prediction_length)):
            if not np.any(np.isnan(model_predictions[:,sample_ind,:,time_ind,:])):
                min_prediction_steps[sample_ind]=time_ind
                break
    def sigma(sim_prof, exp_prof):
        numerator = sim_prof - exp_prof
        numerator = np.square(numerator)
        numerator = np.sum(numerator)
        numerator = numerator / len(exp_prof)
        numerator = np.sqrt(numerator)
        denominator = exp_prof
        denominator = np.square(denominator)
        denominator = np.sum(denominator)
        denominator = denominator / len(exp_prof)
        denominator = np.sqrt(denominator)
        return 100 * (numerator / denominator)
    all_sigmas=np.ones((num_samples,len(model_names),len(recorded_profiles),prediction_length))*np.nan
    for sample_ind in range(num_samples):
        for model_ind,model_name in enumerate(model_names):
            for profile_ind,profile in enumerate(recorded_profiles):
                for time_ind in range(min_prediction_steps[sample_ind]):
                    if profile=='qpsi_EFIT01':
                        all_sigmas[sample_ind,model_ind,profile_ind,time_ind]=sigma(1./model_predictions[model_ind,sample_ind,profile_ind,time_ind],
                                                                                    1./all_info['truth']['data'][sample_ind,profile_ind,time_ind])
                    else:
                        all_sigmas[sample_ind,model_ind,profile_ind,time_ind]=sigma(model_predictions[model_ind,sample_ind,profile_ind,time_ind],
                                                                                    all_info['truth']['data'][sample_ind,profile_ind,time_ind])
    all_sigmas_by_shot=[]
    for shot in np.unique(shots):
        inds=np.where(np.array(shots)==shot)
        if len(inds)>0:
            all_sigmas_by_shot.append(np.nanmean(all_sigmas[inds],axis=0))
    all_sigmas_by_shot=np.array(all_sigmas_by_shot)
    font = {'weight' : 'bold',
            'size'   : 16}
    matplotlib.rc('font', **font)
    legend_fontsize=12
    nan_or_one={model: {profile: 1 for profile in recorded_profiles} for model in model_names}
    nan_profiles={model: [] for model in model_names}
    for model_name in ['astrapredictFIXEDZIPFIT','astrapredictFIXEDGBZIPFIT','astrapredictFIXEDTGLFNNZIPFIT',
                       'astrapredictTGLFNNZIPFIT','astrapredictTGLFNNEPEDNNZIPFIT']:
        nan_profiles[model_name]=['zipfit_edensfit_rho','zeff_rho','qpsi_EFIT01']
    for model_name in ['astrapredictTGLFNNandDiffuseDensityZIPFIT','astrapredictFULLY']:
        nan_profiles[model_name]=['zeff_rho','qpsi_EFIT01']
    for model_name in ['astrapredictFULLYandCurrentZIPFIT']:
        nan_profiles[model_name]=['zeff_rho']
    for model_name in ['astrapredictTGLFNNandCurrentZIPFIT']:
        nan_profiles[model_name]=['zipfit_edensfit_rho','zeff_rho']
    for model_name in model_names:
        for profile in nan_profiles[model_name]:
            nan_or_one[model_name][profile]=np.nan
    change_threshold=500
    changing_actuator='pinj'
    actuator_ind=recorded_actuators.index('pinj')
    #changing_sample_inds=np.where(np.nanstd(all_info['actuator_trajectory']['data'][:,actuator_ind,nwarmup:],axis=-1)>change_threshold)[0]
    changing_sample_inds=np.where(np.abs(np.nanmean(all_info['actuator_trajectory']['data'][:,actuator_ind,nwarmup:nwarmup+10],axis=-1)-np.nanmean(all_info['actuator_trajectory']['data'][:,actuator_ind,-10:],axis=-1))>change_threshold)[0]
    if plot_sigma_bar:
        #legend_fontsize=5
        fig,axes=plt.subplots(len(plotted_profiles),sharex=True,sharey=False,figsize=(10,15))
        axes=np.atleast_1d(axes)
        bar_model_names=[model_name for model_name in model_names if model_name!='const']
        #bar_model_colors=[model_colors[model_name] for model_name in bar_model_names]
        bar_model_labels=[model_name_map.get(model_name,model_name) for model_name in bar_model_names]
        constant_color='b'
        changing_color='r'
        for ax_ind,profile in enumerate(plotted_profiles):
            profile_ind=recorded_profiles.index(profile)
            ax=axes[ax_ind]
            #mean_sigmas,changing_mean_sigmas,std_sigmas,changing_std_sigmas=[],[],[],[]
            sigma_percentiles,changing_sigma_percentiles=[],[]
            mean_sigmas,changing_mean_sigmas=[],[]
            percentiles=[25,50,75]
            for model_name in bar_model_names: #for model_ind,model_name in enumerate(model_names):
                model_ind=model_names.index(model_name)
                changing_sigma_percentiles.append(nan_or_one[model_name][profile]*np.nanpercentile(all_sigmas[changing_sample_inds,model_ind,profile_ind],percentiles))
                sigma_percentiles.append(nan_or_one[model_name][profile]*np.nanpercentile(all_sigmas[:,model_ind,profile_ind],percentiles))
                changing_mean_sigmas.append(nan_or_one[model_name][profile]*np.nanmean(all_sigmas[changing_sample_inds,model_ind,profile_ind]))
                mean_sigmas.append(nan_or_one[model_name][profile]*np.nanmean(all_sigmas[:,model_ind,profile_ind]))
                #changing_std_sigmas.append(nan_or_one[model_name][profile]*np.nanstd(all_sigmas[changing_sample_inds,model_ind,profile_ind]))
                #std_sigmas.append(nan_or_one[model_name][profile]*np.nanstd(all_sigmas[:,model_ind,profile_ind]))
            ind=np.arange(len(bar_model_names))
            width=0.35
            sigma_percentiles=np.array(sigma_percentiles).T
            changing_sigma_percentiles=np.array(changing_sigma_percentiles).T
            sigma_values=sigma_percentiles[1] #mean_sigmas
            changing_sigma_values=changing_sigma_percentiles[1] #changing_mean_sigmas
            rects=ax.bar(ind,sigma_values,width,color=constant_color,alpha=0.8,yerr=(sigma_values-sigma_percentiles[0],
                                                                                     sigma_percentiles[2]-sigma_values))
            ax.set_xticks(ind+width/2)
            ax.set_xticklabels(bar_model_labels,rotation=45)
            const_ind=model_names.index('const')
            #ax.axhline(np.nanmean(all_sigmas[:,const_ind,profile_ind]),c=constant_color,linestyle='--',linewidth=3) #,label='constant on constant')
            # ax.axhspan(np.nanpercentile(all_sigmas[:,const_ind,profile_ind],percentiles[0]),
            #           np.nanpercentile(all_sigmas[:,const_ind,profile_ind],percentiles[2]),
            #           color=constant_color, alpha=0.2,zorder=-100)
            #changing_rects=ax.bar(ind+width,changing_sigma_values,width,alpha=0.8,color=changing_color,yerr=(changing_sigma_values-changing_sigma_percentiles[0],
            #                                                                                                changing_sigma_percentiles[2]-changing_sigma_values))
            #ax.axhline(np.nanmean(all_sigmas[changing_sample_inds,const_ind,profile_ind]),c=changing_color,linestyle='--',linewidth=3) #,
            # ax.axhspan(np.nanpercentile(all_sigmas[changing_sample_inds,const_ind,profile_ind],percentiles[0]),
            #           np.nanpercentile(all_sigmas[changing_sample_inds,const_ind,profile_ind],percentiles[2]),
            #           color=changing_color, alpha=0.2, zorder=-100)
            ax.set_ylabel(sig_name_map.get(profile,profile))
            #ax.set_xticklabels(bar_model_labels, rotation=45)
        # axes[0].legend((rects[0],changing_rects[0]),
        #               ('All trajectories', r'Changing trajectories ($\Delta P_{inj}$>500kW)'),
        #               fontsize=legend_fontsize)
        axes[0].set_title(sigma_bar_title)
        #axes[0].set_ylim(0,30)
        fig.savefig('testbar.png')
        for profile in plotted_profiles:
            profile_ind=recorded_profiles.index(profile)
            baseline_model_ind=1
            baseline_sigma=np.nanmean(all_sigmas_by_shot[:,baseline_model_ind,profile_ind])
            print(profile)
            for model_name in bar_model_names:
                model_ind=model_names.index(model_name)
                if not np.isnan(nan_or_one[model_name][profile]):
                    pval=stats.ttest_rel(np.nanmean(all_sigmas_by_shot[:,baseline_model_ind,profile_ind,:],axis=-1),
                                         np.nanmean(all_sigmas_by_shot[:,model_ind,profile_ind,:],axis=-1),
                                         alternative='greater').pvalue
                    print(f'H: {model_names[baseline_model_ind]}>{model_names[model_ind]}: pvalue={pval}')
    if plot_sigma_time:
        legend_fontsize=20
        fig,axes=plt.subplots(len(plotted_profiles),sharex=True,figsize=(15,10))
        axes=np.atleast_1d(axes)
        dtime=np.arange(1,prediction_length+1)*dataSettings.DT*1.e3
        #time_ind=8
        #my_time=dtime[time_ind]
        for ax_ind,profile in enumerate(plotted_profiles):
            profile_ind=recorded_profiles.index(profile)
            ax=axes[ax_ind]
            for model_ind,model_name in enumerate(model_names):
                # for the plots showing curriculum learning changing num steps over time
                if model_name=='alldiiid_ensembleconfig0EPOCH250':
                    ax.axvline(20,c=model_colors[model_name],linestyle='--')
                if model_name=='alldiiid_ensembleconfig0EPOCH500':
                    ax.axvline(100,c=model_colors[model_name],linestyle='--')
                if model_name=='alldiiid_ensembleconfig0EPOCH750':
                    ax.axvline(200,c=model_colors[model_name],linestyle='--')
                if model_name=='alldiiid_ensembleconfig0':
                    ax.axvline(200,c=model_colors[model_name],linestyle='--')
                mean_sigma=np.nanmean(all_sigmas[:,model_ind,profile_ind],axis=0)
                #bins=np.linspace(0,50,10)
                #ax.hist(all_sigmas[:,model_ind,profile_ind,time_ind],
                #        color=model_colors.get(model_name,'k'),label=model_name,bins=bins,alpha=0.5)
                ax.plot([0]+list(dtime),[0]+list(nan_or_one[model_name][profile]*mean_sigma),
                        c=model_colors.get(model_name,'k'),label=model_name_map.get(model_name,model_name))
            ax.set_xlim(0,None)
            ax.set_ylim(0,None)
            ax.set_ylabel(sig_name_map.get(profile,profile))
        axes[0].legend(fontsize=legend_fontsize,loc='upper left')
        #axes[0].set_title(rf'$\sigma$ error at $\Delta$t={my_time} ms')
        #axes[-1].set_xlabel(r'$\sigma$')
        axes[0].set_title(r'Error (%)')
        axes[-1].set_xlabel(r'$\Delta t$ (ms)')
        fig.savefig('testsigmatime.png')
    # for plots
    #import pdb; pdb.set_trace()
    sample_ind=np.random.choice(changing_sample_inds) #num_samples)
    sample_ind=1479 #meta-learning / ensemble plot
    #sample_ind=486 #20 / 200ms curriculum plot
    shot=shots[sample_ind]
    this_time=int(times[sample_ind])
    if plot_over_rho:
        rho=np.linspace(0,1,dataSettings.nx)
        fig,axes=plt.subplots(1,len(plotted_profiles),sharex=True,figsize=(15,5))
        axes=np.atleast_1d(axes)
        ax_ind=0
        min_prediction_step=min_prediction_steps[sample_ind]-5
        end_time=int(this_time+min_prediction_step*dataSettings.DT*1.e3)
        for profile in plotted_profiles:
            profile_ind=recorded_profiles.index(profile)
            ax=axes[ax_ind]
            ax.plot(rho,all_info['truth']['data'][sample_ind,profile_ind,min_prediction_step,:],linewidth=4,c='k',label='true')
            for model_ind in range(len(model_names)):
                ax.plot(rho,
                        model_predictions[model_ind,sample_ind,profile_ind,min_prediction_step,:],
                        c=model_colors.get(model_names[model_ind],'k'),
                        linestyle=model_linestyles.get(model_names[model_ind],None),
                        label=model_name_map.get(model_names[model_ind],model_names[model_ind]))
            #ax.plot(rho,all_info['profile_warmup']['data'][sample_ind,profile_ind,-1,:],c='k',linestyle='--',label='initial')
            ax.set_ylabel(sig_name_map.get(profile,profile))
            #ax.plot(predicted_times,sim_yhat[sample_ind,profile_ind,:,0],c='b')
            axes[ax_ind].set_xlabel(r'$\rho$')
            ax_ind+=1
        axes[0].legend(fontsize=legend_fontsize)
        fig.suptitle(f'Shot {shot} {this_time}-{end_time}ms')
        axes[-1].set_xlabel(r'$\rho$')
        axes[-1].set_xlim(0,1)
        fig.savefig('testrho.png')
    if plot_over_time:
        #legend_fontsize=20
        fig,axes=plt.subplots(len(plotted_profiles)+len(plotted_actuators),sharex=True,figsize=(10,15))
        axes=np.atleast_1d(axes)
        ax_ind=0
        predicted_times=np.arange(int(times[sample_ind]),
                                  int(times[sample_ind]+prediction_length*dataSettings.DT*1.e3),
                                  dataSettings.DT*1.e3)
        # history times includes the timestep from which the first prediction is made
        present_time=np.array([int(times[sample_ind]-dataSettings.DT*1.e3)])
        history_times=np.arange(int(times[sample_ind]-(nwarmup+1)*dataSettings.DT*1.e3),
                                int(times[sample_ind]-dataSettings.DT*1.e3),
                                dataSettings.DT*1.e3)
        print(f'Shot {shot}, time {times[sample_ind]}, sample index={sample_ind}')
        for profile in plotted_profiles:
            profile_ind=recorded_profiles.index(profile)
            ax=axes[ax_ind]
            ax.plot(np.concatenate([history_times,
                                    present_time,
                                    predicted_times]),
                    np.concatenate([all_info['profile_warmup']['data'][sample_ind,profile_ind,:,0],
                                    all_info['truth']['data'][sample_ind,profile_ind,:,0]]),
                    linewidth=4,
                    label='truth',
                    c='k')
            for model_ind in range(len(model_names)):
                ax.plot(np.concatenate([present_time,
                                        predicted_times]),
                        np.concatenate([[all_info['profile_warmup']['data'][sample_ind,profile_ind,-1,0]],
                                        model_predictions[model_ind,sample_ind,profile_ind,:,0]]),
                        c=model_colors.get(model_names[model_ind],'k'),
                        linestyle=model_linestyles.get(model_names[model_ind],None),
                        label=model_name_map.get(model_names[model_ind],model_names[model_ind]))
            ax.set_ylabel(sig_name_map.get(profile,profile))
            #ax.plot(predicted_times,sim_yhat[sample_ind,profile_ind,:,0],c='b')
            #ax.set_ylim((0,None))
            ax_ind+=1
        for actuator in plotted_actuators:
            actuator_ind=recorded_actuators.index(actuator)
            ax=axes[ax_ind]
            ax.plot(np.concatenate([history_times,present_time,predicted_times]),
                    all_info['actuator_trajectory']['data'][sample_ind,actuator_ind],c='k',
                    linewidth=4)
            ax.set_ylabel(sig_name_map.get(actuator,actuator))
            ax_ind+=1
        axes[0].legend()
        axes[-1].set_xlabel('Time (ms)')
        fig.suptitle(f'Shot {shot}')
        fig.savefig('testtime.png')
    if False:
        fig,axes=plt.subplots(1,figsize=(10,15))
        axes=np.atleast_1d(axes)
        profile='zipfit_etempfit_rho'
        actuator='pinj'
        profile_ind=recorded_profiles.index(profile)
        actuator_ind=recorded_actuators.index(actuator)
        profile_start=np.nanmean(all_info['profile_warmup']['data'][:,profile_ind,:,0],axis=-1)
        true_profile_end=np.nanmean(all_info['truth']['data'][:,profile_ind,-5:,0],axis=-1)
        actuator_changes=np.nanmean(all_info['actuator_trajectory']['data'][:,actuator_ind,-5:],axis=-1)-np.nanmean(all_info['actuator_trajectory']['data'][:,actuator_ind,:nwarmup+1],axis=-1)
        true_profile_changes=true_profile_end-profile_start
        axes[0].scatter(actuator_changes,true_profile_changes,color='k',label='experiment',alpha=0.4)
        for model_ind,model_name in enumerate(model_names):
            if model_name=='const':
                continue
            profile_changes=np.nanmean(model_predictions[model_ind,:,profile_ind,-5:,0],axis=-1)-profile_start
            axes[0].scatter(actuator_changes,profile_changes,color=model_colors.get(model_name,'k'),label=model_name,alpha=0.2)
            axes[0].set_ylabel(sig_name_map.get(profile,profile))
            axes[0].set_xlabel(sig_name_map.get(actuator,actuator))
        axes[0].axvline(0,c='k',linestyle='--')
        axes[0].axhline(0,c='k',linestyle='--')
        axes[0].legend()
        fig.savefig('testcausality.png')
