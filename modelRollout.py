import matplotlib.pyplot as plt
from matplotlib import cm

import configparser
import torch
#torch.manual_seed(0)
from torch.utils.data import DataLoader
import customModels
import customLosses
import customDatasetMakers

import dataSettings
import dataSettings
import numpy as np
import os
import glob
import sys
from customModels import IanRNN, IanMLP, HiroLinear

plot_ensemble=True

models={'IanRNN': IanRNN, 'IanMLP': IanMLP, 'HiroLinear': HiroLinear}
#PRE_STEPS=78
#test_shots=[187070]

label_map={'zipfit_etempfit_rho': r'$T_e$',
           'zipfit_itempfit_rho': r'$T_i$',
           'zipfit_edensfit_rho': r'$n_e$',
           'zipfit_trotfit_rho': r'$\Omega$',
           'qpsi_EFIT01': r'$q$',
           'pinj': r'$P_{NBI}$',
           'tinj': r'$T_{NBI} (N m)$',
           'ip': r'$I_p$',
           'li_EFIT01': 'li',
           'tribot_EFIT01': r'$\delta_l$',
           'tritop_EFIT01': r'$\delta_u$',
           'kappa_EFIT01': r'$\kappa$',
           'volume_EFIT01': 'V',
           'ipsiptargt': r'$I_p^{target}$',
           'dssdenest': r'$<n_e>$'}

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename=f'configs/default.cfg'
config=configparser.ConfigParser()
config.read(config_filename)
output_filename_base=config['model']['output_filename_base']
output_dir=config['model']['output_dir']
model_type=config['model']['model_type']
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs']['parameters'].split()
plotted_profiles=profiles
plotted_actuators=actuators
plotted_parameters=parameters
state_length=len(profiles)*dataSettings.nx+len(parameters)
actuator_length=len(actuators)

data_filename=config['preprocess']['preprocessed_data_filenamebase']+'val.pkl'
#data_filename='/projects/EKOLEMEN/profile_predictor/joe_hiro_models/preprocessed_data_highip_val.pkl'

if plot_ensemble:
    max_loss=0.01
    considered_models=[]
    all_model_files=glob.glob(os.path.join(output_dir, f'{output_filename_base}*.tar'))
    for model_file in all_model_files:
        saved_state=torch.load(model_file, map_location=torch.device('cpu'))
        if saved_state['val_losses'][-1]<max_loss:
            model=models[model_type](input_dim=state_length+2*actuator_length, output_dim=state_length,
                                     **saved_state['model_hyperparams'])
            model.load_state_dict(saved_state['model_state_dict'])
            considered_models.append(model)
    print(f'{len(considered_models)}/{len(all_model_files)} models used (i.e. only loss<{max_loss})')
else:
    model_file=os.path.join(output_dir, f'{output_filename_base}.tar')
    saved_state=torch.load(model_file, map_location=torch.device('cpu'))
    model=models[model_type](input_dim=state_length+2*actuator_length, output_dim=state_length,
                             **saved_state['model_hyperparams'])
    model.load_state_dict(saved_state['model_state_dict'])
    considered_models=[model]

x_test, y_test, shots, times =customDatasetMakers.ian_dataset(data_filename,profiles,actuators,parameters,sort_by_size=True)

#yhat_tensor=model(profiles_tensor, actuators_tensor, parameters_tensor).detach()
#yhat_numpy=yhat_tensor.numpy()
#profiles_numpy=profiles_tensor.detach().numpy()
#for i,profile in enumerate(saved_state['profiles']):
    #yhat_numpy[:,i,:]=dataSettings.denormalize(yhat_numpy[:,i,:], profile)
    #profiles_numpy[:,:,i,:]=dataSettings.denormalize(profiles_numpy[:,:,i,:], profile)
#actuators_numpy=actuators_tensor.detach().numpy()
#for i,actuator in enumerate(saved_state['actuators']):
#    actuators_numpy[:,:,i]=dataSettings.denormalize(actuators_numpy[:,:,i], actuator)

x=np.linspace(0,1,dataSettings.nx)
nrows=max(len(plotted_profiles),len(plotted_actuators))
#fig,axes=plt.subplots(nrows=nrows,ncols=3,sharex='col')#,ncols=2,sharex='col')
#axes=np.atleast_2d(axes)
#axes=axes.T

@torch.no_grad()
class ModelStepper:
    # max_loss determine by e.g. running modelStats.py to see loss over time
    def __init__(self, initial_state, output_dir=output_dir, model_type=model_type, max_loss=0.01):
        num_model_options=0
        self.models=considered_models
        self.all_predictions=torch.zeros((len(self.models),state_length))
        for which_model,model in enumerate(self.models):
            self.all_predictions[which_model]=initial_state
    def prediction_step(self, actuator_array):
        for which_model,model in enumerate(self.models):
            input_tensor=torch.cat((self.all_predictions[which_model],actuator_array))
            self.all_predictions[which_model]=model(input_tensor[None,:])[0]
    def get_denormed_predictions(self):
        normed_dic=dataSettings.state_to_dic(self.all_predictions, profiles=profiles, parameters=parameters)
        for sig in normed_dic:
            normed_dic[sig]=torch.stack(normed_dic[sig])
        return dataSettings.get_denormalized_dic(normed_dic)

sample_ind=0
shot=shots[sample_ind]
start_time=times[sample_ind]
time_length=len(x_test[sample_ind])
predicted_means={}
predicted_stds={}
for profile in plotted_profiles:
    predicted_means[profile]=torch.zeros((time_length,dataSettings.nx))
    predicted_stds[profile]=torch.zeros((time_length,dataSettings.nx))
for parameter in plotted_parameters:
    predicted_means[parameter]=torch.zeros((time_length,1))
    predicted_stds[parameter]=torch.zeros((time_length,1))
#predicted_stds=torch.zeros((time_length,state_length))
for step in range(time_length): #NSTEPS): # loop over predicted timesteps
    #for i in range(lookahead):
    #    profiles_tensor, actuators_tensor, parameters_tensor, extra_sigs_tensor = next(data_iterator)
    step_tensor=x_test[sample_ind][step] #.detach().numpy()
    step_state=step_tensor[:state_length]
    step_actuators=step_tensor[state_length:]
    if step==0:
        modelstepper=ModelStepper(step_state)
    modelstepper.prediction_step(step_actuators)
    denormed_predictions=modelstepper.get_denormed_predictions()
    for sig in plotted_profiles+plotted_parameters:
        predicted_means[sig][step, :]=torch.mean(denormed_predictions[sig], dim=0)
        predicted_stds[sig][step, :]=torch.std(denormed_predictions[sig], dim=0)
times=np.arange(start_time, start_time+time_length*int(dataSettings.DT*1e3), int(dataSettings.DT*1e3))

rho_ind=10
num_columns = 3
if len(plotted_parameters)>0:
    num_columns = 4
fig,axes=plt.subplots(max(len(plotted_profiles),len(plotted_parameters),len(plotted_actuators)),num_columns, sharex='col', figsize=(8,5))
plt.subplots_adjust(hspace=0, wspace=1)
NSTEPS_PLOTTED=4
colors=cm.viridis(np.linspace(0,1,NSTEPS_PLOTTED+1))
plotted_time_inds=[int(t) for t in np.linspace(0, time_length, NSTEPS_PLOTTED, endpoint=False)]
normalized_true_state=x_test[sample_ind]
normalized_true_dic=dataSettings.state_to_dic(normalized_true_state, profiles=profiles, parameters=parameters, actuators=actuators)
for sig in normalized_true_dic:
    normalized_true_dic[sig]=torch.stack(normalized_true_dic[sig])
denormalized_true_dic=dataSettings.get_denormalized_dic(normalized_true_dic)
with torch.no_grad():
    for i,profile in enumerate(plotted_profiles):
        axes[i,0].errorbar(times, predicted_means[profile][:,rho_ind], yerr=predicted_stds[profile][:,rho_ind],
                           label='predicted', c='k', alpha=0.1)
        axes[i,0].plot(times, denormalized_true_dic[profile][:,rho_ind],
                       label='real', c='k', linestyle='--')
        axes[i,0].set_ylabel(label_map[profile])
        for which_color, time_ind in enumerate(plotted_time_inds):
            axes[i,1].plot(x, predicted_means[profile][time_ind], c=colors[which_color],
                           label=f'{times[time_ind]}ms')
            axes[i,1].plot(x, denormalized_true_dic[profile][time_ind],
                           linestyle='--', c=colors[which_color])
    for i,actuator in enumerate(plotted_actuators):
        axes[i,2].plot(times, denormalized_true_dic[actuator],
                       label='real', c='k', linestyle='--')
        axes[i,2].set_ylabel(label_map[actuator])
    if len(plotted_parameters)>0:
        for i,parameter in enumerate(plotted_parameters):
            axes[i,3].errorbar(times, predicted_means[parameter][:, 0], yerr=predicted_stds[parameter][:, 0],
                           label='predicted', c='k', alpha=0.1)
            axes[i,3].plot(times, denormalized_true_dic[parameter],
                           label='real', c='k', linestyle='--')
            axes[i,3].set_ylabel(label_map[parameter])
axes[0,1].legend(fontsize=8)
axes[0,0].legend(fontsize=8)
fig.suptitle(f'Shot {shot}')
plt.savefig(f'{output_filename_base}_plots{sample_ind}.svg', format='svg')
plt.show()
