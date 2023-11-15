import configparser
import torch
#torch.manual_seed(0)
import customDatasetMakers
import dataSettings
import numpy as np
import sys
import os
import prediction_helpers
import plotting_helpers
import pickle
from train_helpers import make_bucket
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from dataSettings import get_denormalized_dic
from customDatasetMakers import state_to_dic

import time

bucket_size=10000
ensemble=False
fake_actuators=False
nwarmup=0

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename=f'configs/default.cfg'
config=configparser.ConfigParser()
config.read(config_filename)
output_filename_base=config['model']['output_filename_base']
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs']['parameters'].split()
plotted_profiles=profiles
plotted_actuators=actuators
plotted_parameters=parameters

appendage=''
if fake_actuators:
    appendage+='_FAKE'
pickle_filename=f"rollout_{output_filename_base}{appendage}.pkl"

data_filename=config['preprocess']['preprocessed_data_filenamebase']+'val.pkl'

x_test, y_test, shots, times =customDatasetMakers.ian_dataset(data_filename,profiles,actuators,parameters,sort_by_size=True)

considered_models = prediction_helpers.get_considered_models(config_filename, ensemble=ensemble)

test_x_buckets = make_bucket(x_test, bucket_size)
test_y_buckets = make_bucket(y_test, bucket_size)
test_length_buckets = [[len(arr) for arr in bucket] for bucket in test_x_buckets]
# used to help index stuff later
running_num_samples=np.insert(np.cumsum([len(bucket) for bucket in test_x_buckets]),0,0)

all_info={}
all_keys=[]
begin_time=time.time()
prev_time=begin_time
for sample_ind in range(len(x_test)):
    if fake_actuators:
        x_test[sample_ind]=prediction_helpers.get_fake_actuator_state(x_test[sample_ind], profiles, parameters, actuators)
    num_times=len(x_test[sample_ind])
    # add 1 because we're looking at predicted compared to target (and -1 for actuator from input step)
    true_times=np.arange(int(times[sample_ind]+dataSettings.DT*1e3),
                         int(times[sample_ind]+(num_times+1)*dataSettings.DT*1e3),
                         dataSettings.DT*1e3)
    start_time=true_times[0]
    end_time=true_times[-1]
    key=f'{shots[sample_ind]}_{int(start_time)}_{int(end_time)}'
    all_keys.append(key)
    all_info[key]={'truth': {'actuators': {}, 'profiles': {}, 'parameters': {}},
                   'predictions': {'profiles': {}, 'parameters': {}}}
    all_info[key]['truth']['times']=np.array(true_times)
    # remember this returns actuators at the present AND NEXT time, hence -1 index below
    input_dic=state_to_dic(x_test[sample_ind], profiles, parameters, actuators)
    denormed_dic=get_denormalized_dic(input_dic)
    for sig in actuators:
        all_info[key]['truth']['actuators'][sig]=denormed_dic[sig][-1]
    output_dic=state_to_dic(y_test[sample_ind], profiles, parameters)
    denormed_dic=get_denormalized_dic(output_dic)
    for sig in profiles:
        all_info[key]['truth']['profiles'][sig]=denormed_dic[sig]
    for sig in parameters:
        all_info[key]['truth']['parameters'][sig]=denormed_dic[sig]
    # predictions start after warmup; right now we just exclude the last timestep
    # but the ground truth for it is in y_test (the targets) if we want it later
    all_info[key]['predictions']['times']=true_times[nwarmup:]
    for sig in parameters:
        all_info[key]['predictions']['parameters'][sig]=np.zeros((len(considered_models),num_times-nwarmup))
    for sig in profiles:
        all_info[key]['predictions']['profiles'][sig]=np.zeros((len(considered_models),num_times-nwarmup,dataSettings.nx))

print(f'Finished writing ground truth, took {time.time()-prev_time:0.0f}s')
evaluation_begin_time=time.time()
prev_time=evaluation_begin_time
with torch.no_grad():
    for which_bucket in range(len(test_x_buckets)):
        x_bucket=test_x_buckets[which_bucket]
        y_bucket=test_y_buckets[which_bucket]
        length_bucket=test_length_buckets[which_bucket]
        padded_x=pad_sequence(x_bucket, batch_first=True)
        #padded_x=padded_x.to(device)
        #padded_y=padded_y.to(device)
        # only save simulations after warmup is over
        for which_model in range(len(considered_models)):
            model=considered_models[which_model]
            model_output=model(padded_x, reset_probability=0, nwarmup=nwarmup)
            unpadded_output=unpad_sequence(model_output, length_bucket, batch_first=True)
            for which_output,output in enumerate(unpadded_output):
                # get the corresponding key
                key=all_keys[running_num_samples[which_bucket]+which_output]
                output_dic=state_to_dic(output, profiles, parameters)
                denormed_dic=get_denormalized_dic(output_dic)
                for sig in denormed_dic:
                    if sig in profiles:
                        all_info[key]['predictions']['profiles'][sig][which_model]=denormed_dic[sig][nwarmup:]
                    elif sig in parameters:
                        all_info[key]['predictions']['parameters'][sig][which_model]=denormed_dic[sig][nwarmup:]
        print(f'Bucket {which_bucket+1}/{len(test_x_buckets)} took {time.time()-prev_time:0.0f}s')
        prev_time=time.time()

print(f"dumping to {pickle_filename}")
with open(pickle_filename,'wb') as f:
    pickle.dump(all_info,f)
print(f"Took {time.time()-begin_time:0.0f}s")
