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
plotted_actuators=['pinj','tinj','dssdenest','ip']

models={'IanRNN': IanRNN, 'IanMLP': IanMLP}
#PRE_STEPS=78
NSTEPS=35
PLOT_STEP=4
#test_shots=[187070]

label_map={'zipfit_etempfit_rho': r'$T_e$',
           'zipfit_itempfit_rho': r'$T_i$',
           'zipfit_edensfit_rho': r'$n_e$',
           'zipfit_trotfit_rho': r'$\Omega$',
           'qpsi_EFIT01': r'$q$',
           'pinj': r'$P_{NBI}$',
           'tinj': r'$T_{NBI} (N m)$',
           'ip': r'$I_p$',
           'dssdenest': r'$<n_e>$'}

model_dir='test_models'
#saved_state=torch.load('test_models/IanRNN.tar', map_location=torch.device('cpu'))
config=configparser.ConfigParser()
config.read(os.path.join(model_dir,'config'))

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
colors=cm.viridis(np.linspace(0,1,NSTEPS+1))

@torch.no_grad()
class ModelStepper:
    # max_loss determine by e.g. running modelStats.py to see loss over time
    def __init__(self, initial_state, model_dir=model_dir, model_type='IanMLP', max_loss=1.0):
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
            import pdb; pdb.set_trace()
            self.all_predictions[which_model]=model(input_tensor)
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
fig,axes=plt.subplots(len(profiles))
with torch.no_grad():
    for i,profile in enumerate(plotted_profiles):
        axes[i].plot(times, [get_denormed_sig_from_state(state, profile)[rho_ind] for state in predicted_means])
plt.show()
# def get_denormed_sig(sig):
#     ind=state_inds[sig]
#     if sig in profiles:
#         tmp=step_state[ind:ind+dataSettings.nx]
#     elif sig in parameters+actuators:
#         tmp=step_state[ind]
#     return dataSettings.denormalize(tmp, sig)
# for i,profile in enumerate(plotted_profiles):
#     profile_ind=state_inds[profile]
#     if step==0:
#         axes[i,0].plot(x, get_denormed_sig(profile),
#                        c=colors[step])
#         axes[i,0].set_ylabel(label_map[profile])
#         axes[i,1].set_ylabel(rf"{label_map[profile]}($\rho=0$)")
#     axes[i,1].scatter(time, get_denormed_sig(profile)[rho_ind],
#                       c=colors[step+1], linestyle='--',marker='o') #,markersize=4)
    # axes[i,1].errorbar(time, predicted_mean[profile_ind][rho_ind],
    #                    predicted_std[profile_ind][rho_ind],
    #                    c=colors[step+1], marker='o')
    # for j in range(5):
    #     axes[i,1].scatter(actuator_times[-1],
    #                       all_predictions_denormed[j][profile_ind][rho_ind],
    #                       color=colors[step+1], marker='o')
#     if step%PLOT_STEP==0:
#         axes[i,0].plot(x,get_denormed_sig(-1, profile_ind, profile),
#                        c=colors[step+1],label=f'{shot}.{time}',linestyle='--')
#         axes[i,0].errorbar(x,
#                            predicted_mean[profile_ind],
#                            #yerr=predicted_std[i],
#                            c=colors[step+1])
# for i,actuator in enumerate(plotted_actuators):
#     actuator_ind=saved_state['actuators'].index(actuator)
#     axes[i,2].plot(actuator_times,
#                    dataSettings.denormalize(actuators_tensor[0,-(lookahead+1):,actuator_ind].detach().numpy(), actuator),
#                    c=colors[step+1],marker='o',markersize=4)
#     axes[i,2].set_ylabel(label_map[actuator])
# axes[-1,0].set_xlabel(r'$\rho$')
# axes[-1,1].set_xlabel('time (ms)')
# axes[-1,2].set_xlabel('time (ms)')
# axes[0,0].legend()
# for ax in np.ndarray.flatten(axes):
#     ax.axhline(0,linestyle='--',c='k')
# plt.show()
#     times.append(recorded_times.detach().numpy())
#     profiles_numpy=profiles_tensor.detach().numpy()
#     output_label=None
#     prediction_label=None
#     normed_prev_step=model(profiles_tensor, actuators_tensor, parameters_tensor).detach()
#     axes[0,i].set_ylabel(profile)
#     axes[0,i].set_xlim(0,1)
#     if 'qpsi' in profile:
#         axes[0,i].set_ylim(0.5,5)
#     if 'etempfit' in profile or 'edensfit' in profile or 'itempfit' in profile or 'pres_' in profile:
#         axes[0,i].set_ylim(0,None)
# axes[0,0].legend()
# for i,actuator in enumerate(saved_state['actuators']):
#     axes[1,i].plot(times,actuators_numpy[batch_ind,:,i])
#     axes[1,i].set_ylabel(actuator)
#     axes[1,i].axvline(present_time,c='k',linestyle='--')


'''
with torch.no_grad():
    val_losses.append(0)
    for *model_inputs, _ in val_loader:
        for i in range(len(model_inputs)):
            model_inputs[i]=model_inputs[i].to(device)
        model_output = model(*model_inputs)
        val_loss = loss_fn(model_output,
                           *model_inputs,
                           profiles, actuators, parameters)
        val_losses[-1]+=val_loss.item()*len(model_inputs[0]) # mean * # samples in batch
'''
'''
with torch.no_grad():
    val_losses.append(0)
    for *model_inputs, _ in val_loader:
        for i in range(len(model_inputs)):
            model_inputs[i]=model_inputs[i].to(device)
        model_output = model(*model_inputs)
        val_loss = loss_fn(model_output,
                           *model_inputs,
                           profiles, actuators, parameters)
        val_losses[-1]+=val_loss.item()*len(model_inputs[0]) # mean * # samples in batch
    val_losses[-1]/=len(val_dataset)
'''
