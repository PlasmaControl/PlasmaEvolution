import configparser
import torch
#torch.manual_seed(0)
import customDatasetMakers
import dataSettings
import numpy as np
import sys
import prediction_helpers
import plotting_helpers

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
#    comparison to the case where use_warmup is True)
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
for sample_ind in range(len(x_test)):
    normalized_true_state=x_test[sample_ind]
    true_times=np.arange(times[sample_ind], times[sample_ind]+len(normalized_true_state)*int(dataSettings.DT*1e3), int(dataSettings.DT*1e3))
    predicted_times=true_times[NUM_WARMUP_STEPS:]
    shot=shots[sample_ind]

    if fake_actuators:
        normalized_true_state=prediction_helpers.get_fake_actuator_state(normalized_true_state, profiles, parameters, actuators)
        appendage+='_Fake'

    predicted_means, predicted_stds = prediction_helpers.get_predictions(normalized_true_state[sim_start_ind:], considered_models, profiles, parameters, nwarmup=nwarmup)
    if use_warmup:
        for profile in profiles:
            predicted_means[profile]=predicted_means[profile][NUM_WARMUP_STEPS:]
            predicted_stds[profile]=predicted_stds[profile][NUM_WARMUP_STEPS:]

    normalized_true_dic=dataSettings.state_to_dic(normalized_true_state, profiles=profiles, parameters=parameters, actuators=actuators)
    for sig in normalized_true_dic:
        normalized_true_dic[sig]=torch.stack(normalized_true_dic[sig])
    denormalized_true_dic=dataSettings.get_denormalized_dic(normalized_true_dic)

    sim_start_time=true_times[NUM_WARMUP_STEPS]
    sim_end_time=true_times[-1]
    title=f'Model_{output_filename_base}_Shot_{shot}.{sim_start_time}to{sim_end_time}{appendage}'
    plotting_helpers.modelRollout_plot(predicted_means, predicted_stds, predicted_times,
                                       denormalized_true_dic, true_times,
                                       plotted_profiles, plotted_parameters, plotted_actuators,
                                       rho_ind, title)
