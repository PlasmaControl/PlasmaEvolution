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

from customModels import IanRNN, IanMLP

plotted_profiles=['zipfit_itempfit_rho','zipfit_edensfit_rho', 'zipfit_etempfit_rho', 'zipfit_trotfit_rho', 'qpsi_EFIT01'] #'zipfit_etempfit_rho'
plotted_actuators=['pinj','tinj','ipsiptargt','dssdenest','ip']
plotted_parameters=['li_EFIT01', 'tribot_EFIT01', 'tritop_EFIT01', 'dssdenest', 'kappa_EFIT01', 'volume_EFIT01']

models={'IanRNN': IanRNN, 'IanMLP': IanMLP}
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

model_dir='test_models'
#saved_state=torch.load('test_models/IanRNN.tar', map_location=torch.device('cpu'))
config=configparser.ConfigParser()
config.read(os.path.join(model_dir,'config2'))

model_type=config['model']['model_type']
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs']['parameters'].split()

#data_filename=config['preprocess']['preprocessed_data_filenamebase']+'test.pkl'
data_filename='preprocessed_data_lowip_test.pkl'

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
state_length=len(profiles)*dataSettings.nx+len(parameters)
actuator_length=len(actuators)
nrows=max(len(plotted_profiles),len(plotted_actuators))
#fig,axes=plt.subplots(nrows=nrows,ncols=3,sharex='col')#,ncols=2,sharex='col')
#axes=np.atleast_2d(axes)
#axes=axes.T

@torch.no_grad()
class ModelStepper:
    # max_loss determine by e.g. running modelStats.py to see loss over time
    def __init__(self, initial_state, model_dir=model_dir, model_type=model_type, max_loss=1.0):
        self.models=[]
        num_model_options=0
        for input_filename in glob.glob(os.path.join(model_dir, f'{model_type}*.tar')):
            saved_state=torch.load(input_filename, map_location=torch.device('cpu'))
            num_model_options+=1
            if saved_state['val_losses'][-1]<max_loss:
                model=models[model_type](input_dim=state_length+2*actuator_length, output_dim=state_length,
                                         **saved_state['model_hyperparams'])
                model.load_state_dict(saved_state['model_state_dict'])
                self.models.append(model)
        print(f'{len(self.models)}/{num_model_options} models used (i.e. only loss<{max_loss})')
        self.all_predictions=torch.zeros((len(self.models),state_length))
        for which_model,model in enumerate(self.models):
            self.all_predictions[which_model]=initial_state
    def prediction_step(self, actuator_array):
        for which_model,model in enumerate(self.models):
            input_tensor=torch.cat((self.all_predictions[which_model],actuator_array))
            #import pdb; pdb.set_trace()
            self.all_predictions[which_model]=model(input_tensor[None,:])[0]
    def get_predictions(self):
        return self.all_predictions

state_inds={}
state_index=0
for profile in profiles:
    state_inds[profile]=state_index
    state_index+=dataSettings.nx
for parameter in parameters:
    state_inds[parameter]=state_index
    state_index+=1
for actuator in actuators:
    state_inds[actuator]=state_index
    state_index+=1
def get_denormed_sig_from_state(state_arr, sig):
    ind=state_inds[sig]
    if sig in profiles:
        tmp=state_arr[ind:ind+dataSettings.nx]
    elif sig in parameters+actuators:
        tmp=state_arr[ind]
    return dataSettings.denormalize(tmp, sig)

sample_ind=0
shot=shots[sample_ind]
start_time=times[sample_ind]
time_length=len(x_test[sample_ind])
predicted_means=torch.zeros((time_length,state_length))
predicted_stds=torch.zeros((time_length,state_length))
for step in range(time_length): #NSTEPS): # loop over predicted timesteps
    #for i in range(lookahead):
    #    profiles_tensor, actuators_tensor, parameters_tensor, extra_sigs_tensor = next(data_iterator)
    step_tensor=x_test[sample_ind][step] #.detach().numpy()
    step_state=step_tensor[:state_length]
    step_actuators=step_tensor[state_length:]
    if step==0:
        modelstepper=ModelStepper(step_state)
    modelstepper.prediction_step(step_actuators)
    all_predictions=modelstepper.get_predictions()
    for profile in plotted_profiles:
        # for now just use the 0th model
        predicted_means[step, :]=all_predictions[0]
    #all_predictions_denormed = modelstepper.get_denormed_predictions()

times=np.arange(start_time, start_time+time_length*int(dataSettings.DT*1e3), int(dataSettings.DT*1e3))
rho_ind=10
fig,axes=plt.subplots(max(len(plotted_profiles),len(plotted_parameters),len(plotted_actuators)),4, sharex='col')

NSTEPS_PLOTTED=4
colors=cm.viridis(np.linspace(0,1,NSTEPS_PLOTTED+1))
plotted_time_inds=[int(t) for t in np.linspace(0, len(predicted_means), NSTEPS_PLOTTED, endpoint=False)]
with torch.no_grad():
    for i,profile in enumerate(plotted_profiles):
        axes[i,0].plot(times, [get_denormed_sig_from_state(state, profile)[rho_ind] for state in predicted_means],
                       label='predicted', c='k')
        axes[i,0].plot(times, [get_denormed_sig_from_state(state, profile)[rho_ind] for state in x_test[sample_ind]],
                       label='real', c='k', linestyle='--')
        axes[i,0].set_ylabel(label_map[profile])
        for which_color, time_ind in enumerate(plotted_time_inds):
            axes[i,1].plot(x, get_denormed_sig_from_state(predicted_means[time_ind], profile), c=colors[which_color],
                           label=f'{times[time_ind]}ms')
            axes[i,1].plot(x, get_denormed_sig_from_state(x_test[sample_ind][time_ind], profile),
                           linestyle='--', c=colors[which_color])
    for i,actuator in enumerate(plotted_actuators):
        axes[i,2].plot(times, [get_denormed_sig_from_state(state, actuator) for state in x_test[sample_ind]],
                       label='real', c='k', linestyle='--')
        axes[i,2].set_ylabel(label_map[actuator])
    for i,parameter in enumerate(plotted_parameters):
        axes[i,3].plot(times, [get_denormed_sig_from_state(state, parameter) for state in predicted_means],
                       label='predicted', c='k')
        axes[i,3].plot(times, [get_denormed_sig_from_state(state, parameter) for state in x_test[sample_ind]],
                       label='real', c='k', linestyle='--')
        axes[i,3].set_ylabel(label_map[parameter])
axes[0,1].legend()
axes[0,0].legend()
fig.suptitle(f'Shot {shot}')
plt.show()
