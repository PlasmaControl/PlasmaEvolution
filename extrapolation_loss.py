import configparser
import torch
import customDatasetMakers
import dataSettings
import numpy as np
import sys
import prediction_helpers
import matplotlib.pyplot as plt

config_list = ['/projects/EKOLEMEN/profile_predictor/joe_hiro_models/YesGasNoDenconfig']
preprocessed_data = '/projects/EKOLEMEN/profile_predictor/joe_hiro_models/preprocessed_diiid_data_highip_val.pkl'

def loss_calculator(configs_list, data_filename):
    loss_dic={}
    for config_filename in configs_list:
        config=configparser.ConfigParser()
        config.read(config_filename)
        output_dir=config['model']['output_dir']
        profiles=config['inputs']['profiles'].split()
        actuators=config['inputs']['actuators'].split()
        parameters=config['inputs']['parameters'].split()

        considered_models = prediction_helpers.get_considered_models(config_filename)
        x_test, y_test, shots, times =customDatasetMakers.ian_dataset(data_filename,profiles,actuators,parameters,sort_by_size=True)
        state_length=len(profiles)*dataSettings.nx+len(parameters)

        loss_per_model_per_signal = np.zeros((len(considered_models), len(profiles+parameters)))

        # shorten considered array
        if len(x_test) > 100:
            x_test = x_test[0::(len(x_test)//100)]
        total_time = np.sum([len(x_test[i]) for i in range(len(x_test))])
        for normalized_true_state in x_test:
            normalized_true_dic=dataSettings.state_to_dic(normalized_true_state, profiles=profiles, parameters=parameters, actuators=actuators)
            for sig in normalized_true_dic:
                normalized_true_dic[sig]=torch.stack(normalized_true_dic[sig])
            denormalized_true_dic=dataSettings.get_denormalized_dic(normalized_true_dic)
            a = prediction_helpers.get_predictions_per_model(normalized_true_state, considered_models, profiles, parameters)
            with torch.no_grad():
                for i, sig in enumerate(profiles+parameters):
                    average_sig_value = np.mean(np.array(denormalized_true_dic[sig]))
                    loss_per_model = [np.sqrt(np.sum(((np.array(a[sig][:,i,:]) - np.array(denormalized_true_dic[sig]))/average_sig_value)**2)) for i in range(len(considered_models))]
                    loss_per_model_per_signal[:,i]+=loss_per_model
        loss_per_model_per_signal = loss_per_model_per_signal/total_time
        loss_dic[config_filename[len(output_dir):]] = [loss_per_model_per_signal, profiles+parameters]
    return loss_dic, data_filename[len(output_dir):-4]

loss_dic, pkl_name= loss_calculator(config_list, preprocessed_data)

import pickle
with open(f'{pkl_name}_losses.pkl', 'wb') as file:
    pickle.dump(loss_dic, file)
