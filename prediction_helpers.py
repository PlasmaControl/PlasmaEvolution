import configparser
import torch
#torch.manual_seed(0)
import dataSettings
import numpy as np
import os
from train_helpers import make_bucket
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
import re
import glob
from customModels import IanRNN, IanMLP, HiroLRAN
from dataSettings import get_denormalized_dic,normalizations
from customDatasetMakers import state_to_dic, dic_to_state
import time

models={'IanRNN': IanRNN, 'IanMLP': IanMLP, 'HiroLRAN': HiroLRAN}

MAX_NUMBER_OF_TIMES=300

# from y_test, get the real profiles at t+1 with warmup removed, equivalent times to ml prediction outputs
def get_ml_truth(y_test,
                 profiles, parameters,
                 recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho'],
                 prediction_length=20, nwarmup=0, use_fancy_normalization=False):
    num_samples=len(y_test)
    num_profiles=len(recorded_profiles)
    # just make this bigger than you think it needs to be
    y=np.ones((num_samples,num_profiles,MAX_NUMBER_OF_TIMES,dataSettings.nx))*np.nan
    for sample_ind in range(num_samples):
        output_dic=state_to_dic(y_test[sample_ind], profiles, parameters)
        #### get input stuff (profile warmup and actuator trajectories
        # only needed for 
        '''input_dic=state_to_dic(x_test[sample_ind], profiles, parameters, calculations, actuators)
        if use_fancy_normalization:
            for actuator in actuators:
                output_dic[actuator]=input_dic[actuator][:,0]'''
        ####
        denormed_dic=get_denormalized_dic(output_dic, use_fancy_normalization=use_fancy_normalization)
        for profile_ind,profile in enumerate(profiles):
            num_times=len(denormed_dic[profile][nwarmup:])
            y[sample_ind,profile_ind,:num_times]=denormed_dic[profile][nwarmup:]
    return y[:,:,:prediction_length]

# get the profiles for warmup times and actuator trajectories from x_test
def get_ml_profile_warmup(x_test,
                          profiles, parameters, calculations, actuators,
                          recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho'],
                          nwarmup=0, use_fancy_normalization=False):
    num_samples=len(x_test)
    profile_warmup=np.ones((num_samples,len(recorded_profiles),nwarmup+1,dataSettings.nx))*np.nan
    for sample_ind in range(num_samples):
        output_dic=state_to_dic(x_test[sample_ind], profiles, parameters, calculations, actuators)
        denormed_dic=get_denormalized_dic(output_dic, use_fancy_normalization=use_fancy_normalization)
        for profile_ind,profile in enumerate(recorded_profiles):
            profile_warmup[sample_ind,profile_ind]=denormed_dic[profile][:nwarmup+1]
    return profile_warmup

# get the actuator trajectory during warmup and prediction times
def get_ml_actuator_trajectory(x_test, 
                               profiles, parameters, calculations, actuators, 
                               prediction_length=20, 
                               nwarmup=0, 
                               use_fancy_normalization=False):
    num_samples=len(x_test)
    actuator_trajectory=np.ones((num_samples,len(actuators),MAX_NUMBER_OF_TIMES))*np.nan
    for sample_ind in range(num_samples):
        output_dic=state_to_dic(x_test[sample_ind], profiles, parameters, calculations, actuators)
        denormed_dic=get_denormalized_dic(output_dic, use_fancy_normalization=use_fancy_normalization)
        for actuator_ind,actuator in enumerate(actuators):
            num_times=len(denormed_dic[actuator])
            actuator_trajectory[sample_ind,actuator_ind,:num_times]=denormed_dic[actuator][:,0]
            actuator_trajectory[sample_ind,actuator_ind,num_times]=denormed_dic[actuator][-1,1]
    return actuator_trajectory[:,:,:prediction_length+nwarmup+1]

# given an x_test, y_test, and a model, outputs the predictions for prediction_length, excluding nwarmup
def get_ml_predictions(x_test, y_test,
                profiles, parameters, calculations, actuators,
                considered_models,
                recorded_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho','zipfit_trotfit_rho'],
                recorded_actuators=['pinj'],
                prediction_length=20,nwarmup=0,
                use_fancy_normalization=False,
                bucket_size=10000):
    test_x_buckets = make_bucket(x_test, bucket_size)
    test_y_buckets = make_bucket(y_test, bucket_size)
    test_length_buckets = [[len(arr) for arr in bucket] for bucket in test_x_buckets]
    num_keys=len(x_test)
    num_profiles=len(recorded_profiles)
    yhat=np.ones((num_keys,num_profiles,MAX_NUMBER_OF_TIMES,dataSettings.nx))*np.nan
    begin_time=time.time()
    prev_time=begin_time
    evaluation_begin_time=time.time()
    prev_time=evaluation_begin_time
    with torch.no_grad():
        sample_ind=0
        for which_bucket in range(len(test_x_buckets)):
            x_bucket=test_x_buckets[which_bucket]
            y_bucket=test_y_buckets[which_bucket]
            length_bucket=torch.tensor(test_length_buckets[which_bucket])
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

def get_ml_profiles_with_warmup(profiles, warmup_ups):
    return np.concatenate((warmup_ups, profiles), axis=2)

# get the predicted profiles at t+1 given the full history, outputs are normalized
def get_fast_profile_prediction(x_test_sample, model):
    with torch.no_grad():
        model_output = model(x_test_sample, reset_probability=1)
    return model_output[:,-1:, :]

def get_considered_models(config_filename, ensemble=True, epoch=None):
    config=configparser.ConfigParser()
    config.read(config_filename)
    output_filename_base=config['model']['output_filename_base']
    output_dir=config['model']['output_dir']
    model_type=config['model']['model_type']
    profiles=config['inputs']['profiles'].split()
    actuators=config['inputs']['actuators'].split()
    parameters=config['inputs']['parameters'].split()
    calculations=config['inputs']['calculations'].split()
    state_length=len(profiles)*dataSettings.nx+len(parameters)
    actuator_length=len(actuators)
    calculation_length=len(calculations)*dataSettings.nx
    considered_models=[]
    epoch_specification=''
    if epoch is not None:
        epoch_specification=f'EPOCH{epoch}'
    if ensemble:
        regex=f'{output_filename_base}[0-9]*{epoch_specification}.tar'
        all_model_files=glob.glob(os.path.join(output_dir, regex))
        # glob is not as powerful, the star is for everything - so whittle down further so it only
        # accepts repeats of numbers
        all_model_files=[model_file for model_file in all_model_files if re.match(regex,os.path.basename(model_file))]
        # exclude models under the median loss
        #losses = []
        #for model_file in all_model_files:
        #    saved_state=torch.load(model_file, map_location=torch.device('cpu'))
        #    losses.append(np.min([saved_state['val_losses'][-i] for i in range(10)]))
        #max_loss = np.median(losses)
        for model_file in all_model_files:
            saved_state=torch.load(model_file, map_location=torch.device('cpu'))
            if True: #np.min([saved_state['val_losses'][-i] for i in range(10)])<max_loss:
                model=models[model_type](input_dim=state_length+calculation_length+2*actuator_length, output_dim=state_length,
                                         **saved_state['model_hyperparams'])
                model.load_state_dict(saved_state['model_state_dict'])
                considered_models.append(model)
        print(f'{len(considered_models)} models used')
        #print(f'{len(considered_models)}/{len(all_model_files)} models used (i.e. only loss<{max_loss:0.2e})')
    else:
        model_file=os.path.join(output_dir, f'{output_filename_base}{epoch_specification}.tar')
        saved_state=torch.load(model_file, map_location=torch.device('cpu'))
        model=models[model_type](input_dim=state_length+calculation_length+2*actuator_length, output_dim=state_length,
                                 **saved_state['model_hyperparams'])
        model.load_state_dict(saved_state['model_state_dict'])
        considered_models=[model]
        print(f'Using {model_file}')
    return considered_models

def get_fake_actuator_state(normalized_true_state, profiles, parameters, actuators):
    fake_actuator_state=normalized_true_state.clone()
    fake_actuator_dic=state_to_dic(fake_actuator_state, profiles=profiles, parameters=parameters, actuators=actuators)
    for actuator in actuators:
        arr = fake_actuator_dic[actuator][0]
        freeze_index = len(arr)//2 - 40
        perturb_index = len(arr)//2 + 30
        perturb_length = 20
        arr[freeze_index:-1] = torch.tensor([arr[freeze_index]]*len(arr[freeze_index:-1]))
        if (actuator=='D_tot'):
            perturb_index = len(arr)//2 + 30
            perturb_length = 30
            arr[perturb_index:perturb_index + perturb_length] = torch.tensor([((np.sin(np.pi*i/perturb_length))*arr[perturb_index] + arr[perturb_index]) for i in range(perturb_length)])
    fake_actuator_state = dic_to_state(fake_actuator_dic, profiles, parameters, actuators=actuators)
    return fake_actuator_state
