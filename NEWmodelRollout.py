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
from dataSettings import get_denormalized_dic
from customDatasetMakers import state_to_dic

import matplotlib.pyplot as plt
import matplotlib

import time

MAX_NUMBER_OF_PREDICTIONS=10000
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

def get_ml_truth(y_test,
                 profiles, parameters,
                 recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho'],
                 prediction_length=20, nwarmup=0, use_fancy_normalization=False):
    num_samples=len(y_test)
    num_profiles=len(profiles)
    # just make this bigger than you think it needs to be
    y=np.ones((num_samples,num_profiles,MAX_NUMBER_OF_TIMES,dataSettings.nx))*np.nan
    for sample_ind in range(num_samples):
        output_dic=state_to_dic(y_test[sample_ind], profiles, parameters)
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
            num_times=len(denormed_dic[actuator][:,0])
            actuator_trajectory[sample_ind,actuator_ind,:num_times]=denormed_dic[actuator][:,0]
            actuator_trajectory[sample_ind,actuator_ind,num_times]=denormed_dic[actuator][-1,1]
    return profile_warmup, actuator_trajectory[:,:,:prediction_length+nwarmup+1]

def get_ml_predictions(x_test,
                       profiles, parameters, calculations, actuators,
                       considered_models,
                       recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho'],
                       prediction_length=15,nwarmup=0,
                       use_fancy_normalization=False,
                       num_rollout_steps=400,
                       bucket_size=10000):
    test_x_buckets = make_bucket(x_test, bucket_size)
    #test_y_buckets = make_bucket(y_test, bucket_size)
    test_length_buckets = [[len(arr) for arr in bucket] for bucket in test_x_buckets]
    # used to help index stuff later
    running_num_samples=np.insert(np.cumsum([len(bucket) for bucket in test_x_buckets]),0,0)
    num_keys=len(x_test)
    num_profiles=len(recorded_profiles)
    yhat=np.ones((num_keys,num_profiles,MAX_NUMBER_OF_PREDICTIONS,dataSettings.nx))*np.nan
    begin_time=time.time()
    prev_time=begin_time
    evaluation_begin_time=time.time()
    prev_time=evaluation_begin_time
    with torch.no_grad():
        sample_ind=0
        for which_bucket in range(len(test_x_buckets)):
            x_bucket=test_x_buckets[which_bucket]
            #y_bucket=test_y_buckets[which_bucket]
            length_bucket=test_length_buckets[which_bucket]
            padded_x=pad_sequence(x_bucket, batch_first=True)
            #padded_x=padded_x.to(device)
            #padded_y=padded_y.to(device)
            # only save simulations after warmup is over
            # see note above, taking out ability to ensemble models
            # since the ethos should be considering different ML and sim
            # models on equal footing
            model=considered_models[0]
            model_output=model(padded_x, reset_probability=0, nwarmup=nwarmup)
            unpadded_output=unpad_sequence(model_output, length_bucket, batch_first=True)
            for which_output,output in enumerate(unpadded_output):
                output_dic=state_to_dic(output, profiles, parameters)
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

def get_sim_predictions_shots_times(sim_name, sim_dir, prediction_length,
                                    recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho'],
                                    min_length=5,
                                    ntimestep_delay=0,
                                    use_delta=False):
    h5_path=os.path.join(sim_dir,sim_name+'.h5')
    name_map={'zipfit_etempfit_rho': 'TE', 'zipfit_itempfit_rho': 'TI', 'zipfit_trotfit_rho': 'UPAR', 'zipfit_edensfit_rho': 'NE', 'qpsi_EFIT01': 'MU',
              'zeff_rho': 'ZEF'}
    recorded_profile_astra_names=[name_map[profile] for profile in recorded_profiles]
    experiment_names={'TE': 'TEX', 'TI': 'TIX', 'UPAR': 'VTORX', 'NE': 'NEX', 'MU': 'MUX', 'ZEF': 'ZEF'}
    yhat=np.ones((MAX_NUMBER_OF_PREDICTIONS,len(recorded_profiles),MAX_NUMBER_OF_TIMES,dataSettings.nx))*np.nan
    with h5py.File(h5_path) as f:
        print('loading h5')
        shots=list(f.keys())
        print('h5 loaded, reading in simulation data')
        key_ind=0
        sim_times=[]
        sim_shots=[]
        for nshot,shot in enumerate(shots):
            # in future might want to make this more lenient, for now force to have the right number of timesteps
            _,indices=extract_chains(f[shot][f'TE_{sim_name}'][:,0],min_length=min_length)
            for indices_index in range(len(indices)):
                # start_index is the time from which the first prediction is made
                # we also save 1 point before this hence +1 throughout
                start_index=indices[indices_index][0]+ntimestep_delay
                end_index=indices[indices_index][1]+1
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
                    #all_info[key]['truth']['profiles'][name_map[predicted_sig]]=f[shot][f'{experiment_names[predicted_sig]}_{sim_name}'][start_index:end_index]
                    if profile=='MU':
                        yhat[key_ind,profile_ind]=1./yhat[key_ind,profile_ind]
                    if profile=='UPAR':
                        upar_scaling=1./(1.e3*f[shot][f'rgeo_{sim_name}'][start_index+1:end_index][:,None])
                        yhat[key_ind,profile_ind,:len(upar_scaling)]=yhat[key_ind,profile_ind,:len(upar_scaling)]*upar_scaling
                key_ind+=1
    print(f'Read in {len(sim_shots)} simulation rollouts')
    return yhat[:key_ind,:,:prediction_length], sim_shots, sim_times

if __name__ == "__main__":
    nwarmup=5
    data_cache_filename='tmp_data.pkl' #'/projects/EKOLEMEN/profile_predictor/final_paper/ip_1000_1200test.pkl'
    raw_data_filename='/projects/EKOLEMEN/profile_predictor/raw_data/diiid_data.h5' #small_test.h5'
    recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho',
                       'zipfit_edensfit_rho', 'qpsi_EFIT01','zeff_rho']
    recorded_actuators=['pinj']
    prediction_length=10
    considered_sims=['astrapredictFULLYtglfnnZIPFIT', 'astrapredictTGLFNNZIPFIT','astrapredictTGLFNNEPEDNNZIPFIT'] #'astrapredictFIXEDZIPFIT','astrapredictFIXEDGBZIPFIT'] #['astrapredictTGLFNNZIPFIT','astrapredictFIXEDZIPFIT','astrapredictFIXEDGBZIPFIT']
    ml_configs=['ip_0_900NOdssdenest_RESUMEDconfig', 'ip_0_1200NOdssdenest_RESUMEDconfig',
                'augip_0_1200NOdssdenestconfig']
    sim_cache_filename='tmp_sim.pkl'
    if not os.path.exists(sim_cache_filename):
        sim_dir="/projects/EKOLEMEN/profile_predictor/sim_data/"
        all_sim_info={}
        for sim_name in considered_sims:
            if sim_name in ['astrapredictTGLFNNZIPFIT','astrapredictTGLFNNEPEDNNZIPFIT','astrapredictFULLYtglfnnZIPFIT']:
                ntimestep_delay=5
                use_delta=True
            else:
                ntimestep_delay=0
                use_delta=False
            sim_yhat, sim_shots, sim_times=get_sim_predictions_shots_times(sim_name, sim_dir, prediction_length=prediction_length,
                                                                           recorded_profiles=recorded_profiles,
                                                                           ntimestep_delay=ntimestep_delay,
                                                                           min_length=15,
                                                                           use_delta=use_delta)
            all_sim_info[sim_name]={'yhat': sim_yhat, 'shots': sim_shots, 'times': sim_times}
        with open(sim_cache_filename,'wb') as f:
            pickle.dump(all_sim_info,f)
    else:
        with open(sim_cache_filename,'rb') as f:
            all_sim_info=pickle.load(f)
    if True:
        time_bounds_to_preprocess=[]
        # jank: right now just use the first sim, in future should take intersection or whatever
        sim_times=all_sim_info[considered_sims[0]]['times']
        shots_to_preprocess=all_sim_info[considered_sims[0]]['shots']
        # ML's first output is 20ms ahead of the start time
        # similarly if we want the last prediction we have to get one extra
        for sample_ind in range(len(sim_times)):
            time_bounds_to_preprocess.append([sim_times[sample_ind]-nwarmup*dataSettings.DT*1.e3,
                                              sim_times[sample_ind]+prediction_length*dataSettings.DT*1.e3])
    if True:
        # get the processed ML data corresponding to the sim times
        profiles=['zipfit_etempfit_rho', 'zipfit_itempfit_rho', 'zipfit_trotfit_rho',
                  'zipfit_edensfit_rho', 'zipfit_zdensfit_rho', 'qpsi_EFIT01']
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
                                                ip_minimum=1.0e6,ip_maximum=1.2e6,
                                                zero_fill_signals=['ech_pwr_total','pinj','tinj'])
    # now get the models and dump the predictions
    ml_cache_filename='tmp_ml.pkl'
    if not os.path.exists(ml_cache_filename):
        ml_model_dirname='/projects/EKOLEMEN/profile_predictor/final_paper_models/'
        all_ml_info={}
        for ml_config in ml_configs:
            config_filename=os.path.join(ml_model_dirname, ml_config)
            config=configparser.ConfigParser()
            config.read(config_filename)
            profiles=config['inputs']['profiles'].split()
            actuators=config['inputs']['actuators'].split()
            parameters=config['inputs'].get('parameters','').split()
            calculations=config['inputs'].get('calculations','').split()
            use_fancy_normalization=config['preprocess'].getboolean('use_fancy_normalization',False)
            ensemble=False
            fake_actuators=False
            epoch=None
            num_rollout_steps=400
            min_sample_length=13 #num_rollout_steps+nwarmup
            x_test, y_test, ml_shots, times =customDatasetMakers.ian_dataset(data_cache_filename,profiles,parameters,calculations,actuators,sort_by_size=True,
                                                                             min_sample_length=min_sample_length,
                                                                             use_fancy_normalization=use_fancy_normalization)
            ml_times=np.array(times)+nwarmup*dataSettings.DT*1.e3
            ml_times=ml_times.astype(int)
            start_times=ml_times
            # ml prediction stuff
            considered_models = prediction_helpers.get_considered_models(config_filename, ensemble=ensemble, epoch=epoch)
            ml_predictions=get_ml_predictions(x_test,
                                              profiles, parameters, calculations, actuators,
                                              considered_models,
                                              recorded_profiles=recorded_profiles,
                                              prediction_length=prediction_length,
                                              nwarmup=nwarmup, use_fancy_normalization=use_fancy_normalization,
                                              num_rollout_steps=num_rollout_steps)
            all_ml_info[ml_config]={'yhat': ml_predictions, 'shots': ml_shots, 'times': ml_times}
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
        truth=get_ml_truth(y_test,
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
    all_model_info={}
    all_model_info.update(all_sim_info)
    all_model_info.update(all_ml_info)
    shot_time_info={}
    shot_time_info['truth']={'shots': ml_shots, 'times': ml_times}
    for model in all_model_info:
        shot_time_info[model]={'shots': all_model_info[model]['shots'], 'times': all_model_info[model]['times']}
    shot_time_keys={signal: [f"{shot_time_info[signal]['shots'][ind]}_{shot_time_info[signal]['times'][ind]}"
                             for ind in range(len(shot_time_info[signal]['shots']))]
                    for signal in shot_time_info}
    shared_keys=sorted(list(set.intersection(*[set(shot_time_keys[signal]) for signal in shot_time_info])))
    shots=[int(key.split('_')[0]) for key in shared_keys]
    times=[int(key.split('_')[1]) for key in shared_keys]
    new_indices={signal: [shot_time_keys[signal].index(key) for key in shared_keys] for signal in shot_time_keys}
    for model in all_model_info:
        all_model_info[model]['shots']=shots
        all_model_info[model]['times']=times
        all_model_info[model]['yhat']=all_model_info[model]['yhat'][new_indices[model]]
    truth=truth[new_indices['truth']]
    profile_warmup=profile_warmup[new_indices['truth']]
    actuator_trajectory=actuator_trajectory[new_indices['truth']]
    num_samples=len(shots)

    models_for_blend=['ip_0_900NOdssdenest_RESUMEDconfig', *considered_sims]
    blended_predictions=np.sum([all_model_info[model]['yhat'] for model in models_for_blend],axis=0)/len(models_for_blend)
    const_predictions=np.ones((num_samples,len(recorded_profiles),prediction_length,dataSettings.nx))
    for time_ind in range(const_predictions.shape[-2]):
        const_predictions[:,:,time_ind,:]=profile_warmup[:,:,-1,:]
    model_predictions=np.stack([*[all_model_info[model]['yhat'] for model in considered_sims],
                                *[all_model_info[model]['yhat'] for model in ml_configs],
                                blended_predictions,
                                const_predictions])
    model_names=[*considered_sims,*ml_configs,'blended','const']
    min_prediction_steps=np.zeros(num_samples).astype(int)
    for sample_ind in range(num_samples):
        for time_ind in reversed(range(prediction_length)):
            if not np.any(np.isnan(model_predictions[:,sample_ind,:,time_ind,:])):
                min_prediction_steps[sample_ind]=time_ind
                break

    model_colors={'ip_0_1200NOdssdenest_RESUMEDconfig': 'c', 'ip_0_900NOdssdenest_RESUMEDconfig': 'r',
                  'astrapredictTGLFNNZIPFIT': 'b', 'astrapredictFIXEDZIPFIT': 'm', 'astrapredictFIXEDGBZIPFIT': 'b',
                  'astrapredictFIXEDTGLFNNZIPFIT': 'm',
                  'blended': 'g', 'const': 'k'}
    model_linestyles={'const': '--'} #, 'ip_0_1200NOdssdenest_RESUMEDconfig': '--'}
    sim_color_map=matplotlib.colormaps['winter'](np.linspace(0,1,len(considered_sims)))
    ml_color_map=matplotlib.colormaps['autumn'](np.linspace(0,1,len(ml_configs)))
    for i,model_name in enumerate(considered_sims):
        model_colors[model_name]=sim_color_map[i]
    for i,model_name in enumerate(ml_configs):
        model_colors[model_name]=ml_color_map[i]
    model_name_map={'const': 'initial',
                    'ip_0_1200NOdssdenest_RESUMEDconfig': 'ip_1200kA', 'ip_0_900NOdssdenest_RESUMEDconfig': 'ip_900kA',
                    'astrapredictTGLFNNZIPFIT': 'tglfnn', 'astrapredictFIXEDZIPFIT': 'fixed', 'astrapredictFIXEDGBZIPFIT': 'fixedGB',
                    'astrapredictFIXEDTGLFNNZIPFIT': 'fixed (tglfnn-adjusted)',
                    'astrapredictTGLFNNEPEDNNZIPFIT': 'tglfnn+epednn',
                    'astrapredictFULLYtglfnnZIPFIT': 'tglfnn+epednn+ne+Vtor',
                    'astrapredictTGLFNNZIPFIT': 'tglfnn)'}
    sig_name_map={'zipfit_etempfit_rho': r'$T_e$', 'zipfit_itempfit_rho': r'$T_i$', 'zipfit_trotfit_rho': r'$\Omega$',
                  'zipfit_edensfit_rho': r'$n_e$', 'zeff_rho': r'$Z_{eff}$', 'qpsi_EFIT01': 'q',
                  'pinj': r'$P_{inj}$', 'ip': r'$I_p$'}

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
                    all_sigmas[sample_ind,model_ind,profile_ind,time_ind]=sigma(model_predictions[model_ind,sample_ind,profile_ind,time_ind],
                                                                                truth[sample_ind,profile_ind,time_ind])
    font = {'weight' : 'bold',
            'size'   : 16}
    matplotlib.rc('font', **font)
    legend_fontsize=10
    if True:
        fig,axes=plt.subplots(len(recorded_profiles),sharex=True,figsize=(10,15))
        axes=np.atleast_1d(axes)
        dtime=np.arange(prediction_length)*dataSettings.DT*1.e3
        time_ind=8
        my_time=dtime[time_ind]
        for profile_ind,profile in enumerate(recorded_profiles):
            ax=axes[profile_ind]
            for model_ind,model_name in enumerate(model_names):
                mean_sigma=np.nanmedian(all_sigmas[:,model_ind,profile_ind],axis=0)
                #bins=np.linspace(0,50,10)
                #ax.hist(all_sigmas[:,model_ind,profile_ind,time_ind],
                #        color=model_colors.get(model_name,'k'),label=model_name,bins=bins,alpha=0.5)
                ax.plot(dtime,mean_sigma,c=model_colors[model_name],label=model_name)
            ax.set_ylabel(sig_name_map.get(profile,profile))
        axes[0].legend(fontsize=legend_fontsize)
        #axes[0].set_title(rf'$\sigma$ error at $\Delta$t={my_time} ms')
        #axes[-1].set_xlabel(r'$\sigma$')
        axes[0].set_title(r'$\sigma$ error')
        axes[-1].set_xlabel(r'$\Delta t$')
        fig.savefig('testsigma.png')
    # for plots
    sample_ind=9 #np.random.choice(num_samples)
    shot=shots[sample_ind]
    this_time=int(times[sample_ind])
    if True:
        rho=np.linspace(0,1,dataSettings.nx)
        fig,axes=plt.subplots(len(recorded_profiles),sharex=True,figsize=(10,15))
        axes=np.atleast_1d(axes)
        ax_ind=0
        min_prediction_step=min_prediction_steps[sample_ind]
        end_time=int(this_time+min_prediction_step*dataSettings.DT*1.e3)
        for profile_ind,profile in enumerate(recorded_profiles):
            ax=axes[ax_ind]
            for model_ind in range(len(model_names)):
                ax.plot(rho,
                        model_predictions[model_ind,sample_ind,profile_ind,min_prediction_step,:],
                        c=model_colors.get(model_names[model_ind],'k'),
                        linestyle=model_linestyles.get(model_names[model_ind],None),
                        label=model_name_map.get(model_names[model_ind],model_names[model_ind]))
            #ax.plot(rho,profile_warmup[sample_ind,profile_ind,-1,:],c='k',linestyle='--',label='initial')
            ax.plot(rho,truth[sample_ind,profile_ind,min_prediction_step,:],c='k',label='true')
            ax.set_ylabel(sig_name_map.get(profile,profile))
            #ax.plot(predicted_times,sim_yhat[sample_ind,profile_ind,:,0],c='b')
            ax_ind+=1
        axes[0].legend(fontsize=legend_fontsize)
        axes[0].set_title(f'Shot {shot} {this_time}-{end_time}ms')
        axes[-1].set_xlabel(r'$\rho$')
        axes[-1].set_xlim(0,1)
        fig.savefig('testrho.png')
    if True:
        fig,axes=plt.subplots(len(recorded_profiles)+len(recorded_actuators),sharex=True,figsize=(10,15))
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
        for profile_ind,profile in enumerate(recorded_profiles):
            ax=axes[ax_ind]
            for model_ind in range(len(model_names)):
                ax.plot(np.concatenate([present_time,
                                        predicted_times]),
                        np.concatenate([[profile_warmup[sample_ind,profile_ind,-1,0]],
                                        model_predictions[model_ind,sample_ind,profile_ind,:,0]]),
                        c=model_colors.get(model_names[model_ind],'k'),
                        linestyle=model_linestyles.get(model_names[model_ind],None),
                        label=model_name_map.get(model_names[model_ind],model_names[model_ind]))
            ax.plot(np.concatenate([history_times,
                                    present_time,
                                    predicted_times]),
                    np.concatenate([profile_warmup[sample_ind,profile_ind,:,0],
                                    truth[sample_ind,profile_ind,:,0]]),
                    c='k')
            ax.set_ylabel(sig_name_map.get(profile,profile))
            #ax.plot(predicted_times,sim_yhat[sample_ind,profile_ind,:,0],c='b')
            ax.set_ylim((0,None))
            ax_ind+=1
        for actuator_ind,actuator in enumerate(recorded_actuators):
            ax=axes[ax_ind]
            ax.plot(np.concatenate([history_times,present_time,predicted_times]),
                    actuator_trajectory[sample_ind,actuator_ind],c='k')
            ax.set_ylabel(sig_name_map.get(actuator,actuator))
            ax_ind+=1
        axes[0].legend(fontsize=legend_fontsize)
        axes[-1].set_xlabel('Time (s)')
        axes[0].set_title(f'Shot {shot}')
        fig.savefig('testtime.png')
    if False:
        fig,axes=plt.subplots(len(recorded_profiles),sharex=True)
        axes=np.atleast_1d(axes)
        dtime=(np.arange(prediction_length)+1)*dataSettings.DT*1.e3
        for profile_ind,profile in enumerate(recorded_profiles):
            ax=axes[profile_ind]
            for model_ind,model_name in enumerate(model_names):
                mean_sigma=np.nanmedian(all_sigmas[:,model_ind,profile_ind],axis=0)
                ax.plot(dtime,mean_sigma,c=model_colors.get(model_name,'k'),label=model_name_map.get(model_name,model_name))
            ax.set_ylabel(sig_name_map[profile])
        axes[0].legend()
        axes[0].set_title(r'$\sigma$ error')
        axes[-1].set_xlabel(r'$\Delta$t (ms)')
        fig.savefig('testsigma.png')
