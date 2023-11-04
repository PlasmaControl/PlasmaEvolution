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
fake_actuators = False
sample_ind=0
rho_ind = 10

x_test, y_test, shots, times =customDatasetMakers.ian_dataset(data_filename,profiles,actuators,parameters,sort_by_size=True)

considered_models = prediction_helpers.get_considered_models(config_filename, plot_ensemble)

normalized_true_state=x_test[sample_ind]
shot=shots[sample_ind]
start_time=times[sample_ind]
time_length=len(normalized_true_state)
plotted_times=np.arange(start_time, start_time+time_length*int(dataSettings.DT*1e3), int(dataSettings.DT*1e3))
title=f'Model_{output_filename_base}_Shot_{shot}'

if fake_actuators:
    normalized_true_state=prediction_helpers.get_fake_actuator_state(normalized_true_state, profiles, parameters, actuators)
    title+='_Fake'
predicted_means, predicted_stds = prediction_helpers.get_predictions(normalized_true_state, considered_models, profiles, parameters)

normalized_true_dic=dataSettings.state_to_dic(normalized_true_state, profiles=profiles, parameters=parameters, actuators=actuators)
for sig in normalized_true_dic:
    normalized_true_dic[sig]=torch.stack(normalized_true_dic[sig])
denormalized_true_dic=dataSettings.get_denormalized_dic(normalized_true_dic)

plotting_helpers.modelRollout_plot(predicted_means, predicted_stds, denormalized_true_dic, plotted_times, time_length, plotted_profiles, plotted_parameters, plotted_actuators, rho_ind, shot, title)
