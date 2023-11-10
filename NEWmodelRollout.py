import configparser
import torch
#torch.manual_seed(0)
import customDatasetMakers
import dataSettings
import numpy as np
import sys
import prediction_helpers
import plotting_helpers
import pickle
from train_helpers import make_bucket
from torch.nn.utils.rnn import pad_sequence, unpad_sequence
from dataSettings import state_to_dic, get_denormalized_dic

pickle_filename='dumped_predictions.pkl'
bucket_size=50
ensemble=True
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

data_filename=config['preprocess']['preprocessed_data_filenamebase']+'train.pkl'

x_test, y_test, shots, times =customDatasetMakers.ian_dataset(data_filename,profiles,actuators,parameters,sort_by_size=True)

considered_models = prediction_helpers.get_considered_models(config_filename, ensemble=ensemble)

test_x_buckets = make_bucket(x_test, bucket_size)
test_y_buckets = make_bucket(y_test, bucket_size)
test_length_buckets = [[len(arr) for arr in bucket] for bucket in test_x_buckets]
# used to help index stuff later
running_num_samples=np.insert(np.cumsum([len(bucket) for bucket in test_x_buckets]),0,0)

all_info={}
all_keys=[]
for sample_ind in range(len(x_test)):
    num_times=len(x_test[sample_ind])
    true_times=np.arange(times[sample_ind], times[sample_ind]+num_times*int(dataSettings.DT*1e3), int(dataSettings.DT*1e3))
    start_time=true_times[0]
    end_time=true_times[-1]
    key=f'{shots[sample_ind]}_{start_time}_{end_time}'
    all_keys.append(key)
    all_info[key]={}
    all_info[key]['times']=true_times
    input_dic=state_to_dic(x_test[sample_ind], profiles, parameters, actuators)
    denormed_dic=get_denormalized_dic(input_dic)
    for sig in actuators:
        all_info[key][sig]=denormed_dic[sig]
    output_dic=state_to_dic(y_test[sample_ind], profiles, parameters)
    denormed_dic=get_denormalized_dic(output_dic)
    for sig in profiles+parameters:
        all_info[key][sig]=denormed_dic[sig]
    # predictions start after warmup; right now we just exclude the last timestep
    # but the ground truth for it is in y_test (the targets) if we want it later
    all_info[key]['predicted_times']=true_times[nwarmup:]
    for sig in parameters:
        all_info[key][f'predicted_{sig}']=np.zeros((len(considered_models),num_times-nwarmup))
    for sig in profiles:
        all_info[key][f'predicted_{sig}']=np.zeros((len(considered_models),num_times-nwarmup,dataSettings.nx))
        
with torch.no_grad():
    for which_bucket in range(len(test_x_buckets)):
        x_bucket=test_x_buckets[which_bucket]
        y_bucket=test_y_buckets[which_bucket]
        length_bucket=test_length_buckets[which_bucket]
        padded_x=pad_sequence(x_bucket, batch_first=True)
        #padded_x=padded_x.to(device)
        #padded_y=padded_y.to(device)
        # only save simulations after warmup is over
        for i in range(len(considered_models)):
            model=considered_models[i]
            model_output=model(padded_x, autoregression_probability=1, nwarmup=nwarmup)[:,nwarmup:,:]
            unpadded_output=unpad_sequence(model_output, length_bucket, batch_first=True)
            for which_output,output in enumerate(unpadded_output):
                # get the corresponding key
                key=all_keys[running_num_samples[which_bucket]+which_output]
                output_dic=state_to_dic(output, profiles, parameters)
                denormed_dic=get_denormalized_dic(output_dic)
                for sig in denormed_dic:
                    all_info[key][f'predicted_{sig}'][i]=denormed_dic[sig]

with open(pickle_filename,'wb') as f:
    pickle.dump(all_info,f)
