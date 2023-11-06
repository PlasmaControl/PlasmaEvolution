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

plot=True
dump_pickle=False
pickle_filename='dumped_predictions.pkl'

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

data_filename=config['preprocess']['preprocessed_data_filenamebase']+'val.pkl'
#data_filename='/projects/EKOLEMEN/profile_predictor/joe_hiro_models/preprocessed_data_highip_val.pkl'
plot_ensemble=True
fake_actuators=False
sample_ind=1
rho_ind=10
# if use_warmup is True, number of steps to warmup on
#    if False, number of steps to plot from the truth but not from the prediction (just to give apples-to-apples
#    comparison to the case where use_warmup is True; and also can be used to easily change where predictions start)
NUM_WARMUP_STEPS=0
# If num_warmup_steps is 0, set the below to false so it doesn't name the file with WARMUP (would be confusing)
use_warmup=False

x_test, y_test, shots, times =customDatasetMakers.ian_dataset(data_filename,profiles,actuators,parameters,sort_by_size=True)

considered_models = prediction_helpers.get_considered_models(config_filename, plot_ensemble)

if use_warmup:
    nwarmup=NUM_WARMUP_STEPS
    sim_start_ind=0
    appendage='_WARMUP'
else:
    nwarmup=0
    sim_start_ind=NUM_WARMUP_STEPS
    appendage=''
if dump_pickle:
    dump_data={}
for sample_ind in range(len(x_test)):
    normalized_true_state=x_test[sample_ind]
    true_times=np.arange(times[sample_ind], times[sample_ind]+len(normalized_true_state)*int(dataSettings.DT*1e3), int(dataSettings.DT*1e3))
    shot=shots[sample_ind]
    if len(x_test[sample_ind])<=NUM_WARMUP_STEPS:
        print(f"{shot} doesn't have enough timesteps, going to next shot")
        continue
    print(shot)

    if fake_actuators:
        normalized_true_state=prediction_helpers.get_fake_actuator_state(normalized_true_state, profiles, parameters, actuators)
        appendage+='_Fake'

    predicted_means, predicted_stds = prediction_helpers.get_predictions(normalized_true_state[sim_start_ind:], considered_models, profiles, parameters, nwarmup=nwarmup)

    normalized_true_dic=dataSettings.state_to_dic(normalized_true_state, profiles=profiles, parameters=parameters, actuators=actuators)
    for sig in normalized_true_dic:
        normalized_true_dic[sig]=torch.stack(normalized_true_dic[sig])
    denormalized_true_dic=dataSettings.get_denormalized_dic(normalized_true_dic)
    for sig in denormalized_true_dic:
        denormalized_true_dic[sig]=denormalized_true_dic[sig].detach().numpy()
    sim_start_time=true_times[NUM_WARMUP_STEPS]
    sim_end_time=true_times[-1]

    if plot:
        title=f'Model_{output_filename_base}_Shot_{shot}.{sim_start_time}to{sim_end_time}{appendage}'
        predicted_times_plot=true_times[NUM_WARMUP_STEPS:]
        predicted_means_plot={profile: predicted_means[profile][NUM_WARMUP_STEPS:] for profile in profiles}
        predicted_stds_plot={profile: predicted_stds[profile][NUM_WARMUP_STEPS:] for profile in profiles}
        plotting_helpers.modelRollout_plot(predicted_means_plot, predicted_stds_plot, predicted_times_plot,
                                           denormalized_true_dic, true_times,
                                           plotted_profiles, plotted_parameters, plotted_actuators,
                                           rho_ind, title)
    if dump_pickle:
        dump_data[shot]={}
        dump_data[shot]['times']=true_times
        for sig in plotted_profiles+plotted_parameters:
            dump_data[shot][sig]=predicted_means[sig]
        for actuator in actuators:
            dump_data[shot][actuator]=denormalized_true_dic[actuator]
if dump_pickle:
    with open(pickle_filename, 'wb') as f:
        pickle.dump(dump_data,f)
