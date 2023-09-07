import matplotlib.pyplot as plt
from matplotlib import cm

import configparser
import torch
#torch.manual_seed(0)
from torch.utils.data import DataLoader
import customModels
import customLosses
import customDatasetMakers

from dataSettings import nx, test_shots
import dataSettings
import numpy as np
import os

plotted_profiles=['zipfit_itempfit_rho','zipfit_edensfit_rho', 'zipfit_etempfit_rho', 'zipfit_trotfit_rho', 'qpsi_EFIT01'] #'zipfit_etempfit_rho'
plotted_actuators=['pinj','tinj','dssdenest','ip']

PRE_STEPS=78
NSTEPS=35
PLOT_STEP=4
test_shots=[187070]

label_map={'zipfit_etempfit_rho': r'$T_e$',
           'zipfit_itempfit_rho': r'$T_i$',
           'zipfit_edensfit_rho': r'$n_e$',
           'zipfit_trotfit_rho': r'$\Omega$',
           'qpsi_EFIT01': r'$q$',
           'pinj': r'$P_{NBI}$',
           'tinj': r'$T_{NBI} (N m)$',
           'ip': r'$I_p$',
           'dssdenest': r'$<n_e>$'}

model_dir='models2lookahead'
saved_state=torch.load('models2lookahead/PlasmaConv2D0.tar', map_location=torch.device('cpu'))
config=configparser.ConfigParser()
config.read(os.path.join(model_dir,'config10'))
data_filename='test.h5'
ip_minimum=float(config['data']['ip_minimum'])
ip_maximum=float(config['data']['ip_maximum'])
lookahead=int(config['inputs']['lookahead'])
lookback=int(config['inputs']['lookback'])
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs']['parameters'].split()
space_inds=[int(key) for key in config['inputs']['space_inds'].split()]
if len(space_inds)==0:
    space_inds=list(range(nx))
datasetParams={'lookahead': lookahead, 'lookback': lookback,
               'space_inds': space_inds, 'ip_minimum': ip_minimum, 'ip_maximum': ip_maximum}
#test_shots=test_shots[-1000:-970]
test_dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,shots=test_shots,**datasetParams)

data_loader=DataLoader(test_dataset, batch_size=1)

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
fig,axes=plt.subplots(nrows=nrows,ncols=3,sharex='col')#,ncols=2,sharex='col')
axes=np.atleast_2d(axes)
#axes=axes.T
colors=cm.viridis(np.linspace(0,1,NSTEPS+1))

class ModelStepper:
    # max_loss determine by e.g. running modelStats.py to see loss over time
    def __init__(self, initial_profiles, model_dir=model_dir, num_models=100, max_loss=0.3):
        self.models=[]
        for i in range(num_models):
            input_filename=os.path.join(model_dir,f'PlasmaConv2D{i}.tar')
            saved_state=torch.load(input_filename, map_location=torch.device('cpu'))
            if saved_state['val_losses'][-1]<max_loss:
                model=customModels.PlasmaConv2D(saved_state['profiles'], saved_state['actuators'], saved_state['parameters'])
                model.load_state_dict(saved_state['model_state_dict'])
                self.models.append(model)
        print(f'{len(self.models)}/{num_models} models used (i.e. only loss<{max_loss})')
        self.all_predictions=np.zeros((len(self.models),len(profiles),dataSettings.nx))
        for which_model,model in enumerate(self.models):
            self.all_predictions[which_model]=initial_profiles
    def prediction_step(self, actuators_tensor, parameters_tensor):
        for which_model,model in enumerate(self.models):
            # right now we take in all profiles into the past, though in practice only use last one
            # so here we just make a tensor of the proper shape by repeating lookback times
            prev_profile_tensor=torch.Tensor(self.all_predictions[which_model]).unsqueeze(0).repeat(lookback,1,1).unsqueeze(0)
            self.all_predictions[which_model]=model(prev_profile_tensor, actuators_tensor, parameters_tensor).detach().numpy()
    def get_denormed_predictions(self):
        all_predictions_denormed=np.zeros_like(self.all_predictions)
        for model_ind in range(len(self.models)):
            for profile_ind,profile in enumerate(profiles):
                all_predictions_denormed[model_ind][profile_ind]=dataSettings.denormalize(self.all_predictions[model_ind][profile_ind],profile)
        return all_predictions_denormed
    def get_mean_and_std(self):
        all_predictions_denormed=self.get_denormed_predictions()
        #means=np.mean(all_predictions_denormed,axis=0)
        #stds=np.std(all_predictions_denormed,axis=0)
        means=np.median(all_predictions_denormed,axis=0)
        stds=np.subtract(*np.percentile(all_predictions_denormed, [75, 25],axis=0))
        return means, stds

data_iterator=iter(data_loader)
for i in range(PRE_STEPS):
    next(data_iterator)

rho_ind=0
for step in range(NSTEPS): # loop over predicted timesteps
    for i in range(lookahead):
        profiles_tensor, actuators_tensor, parameters_tensor, extra_sigs_tensor = next(data_iterator)
    if step==0:
        modelstepper=ModelStepper(profiles_tensor[:,0,:,:].detach().numpy())
        #modelstepper.prediction_initialize(profiles_tensor, actuators_tensor, parameters_tensor)
        #predicted_profiles_tensor=profiles_tensor
    modelstepper.prediction_step(actuators_tensor, parameters_tensor)
    predicted_mean, predicted_std = modelstepper.get_mean_and_std()
    all_predictions_denormed = modelstepper.get_denormed_predictions()
    #predicted_mean, predicted_std, normed_mean=run_ensemble(predicted_profiles_tensor, actuators_tensor, parameters_tensor)

    def get_true_profile(time_ind, profile_ind, profile):
        return dataSettings.denormalize(profiles_tensor[0,time_ind,profile_ind,:].detach().numpy(), profile)

    shot=int(extra_sigs_tensor[:,0])
    time=int(extra_sigs_tensor[:,1])
    DT_milliseconds=int(dataSettings.DT*1e3)
    actuator_times=np.arange(time,
                             time+DT_milliseconds*(lookahead+1),
                             DT_milliseconds)
    for i,profile in enumerate(plotted_profiles):
        profile_ind=saved_state['profiles'].index(profile)
        if step==0:
            axes[i,0].plot(x, get_true_profile(0, profile_ind, profile),
                           c=colors[step],label=f'{shot}.{time-DT_milliseconds*lookahead} (initial)',linestyle=':')
            axes[i,0].set_ylabel(label_map[profile])
            axes[i,1].set_ylabel(rf"{label_map[profile]}($\rho=0$)")
        axes[i,1].plot(actuator_times,
                       [get_true_profile(time_ind, profile_ind, profile)[rho_ind] for time_ind in range(len(actuator_times))],
                       c=colors[step+1], linestyle='--',marker='o',markersize=4)
        axes[i,1].errorbar(actuator_times[-1],
                           predicted_mean[profile_ind][rho_ind],
                           predicted_std[profile_ind][rho_ind],
                           c=colors[step+1], marker='o')
        # for j in range(5):
        #     axes[i,1].scatter(actuator_times[-1],
        #                       all_predictions_denormed[j][profile_ind][rho_ind],
        #                       color=colors[step+1], marker='o')
        if step%PLOT_STEP==0:
            axes[i,0].plot(x,get_true_profile(-1, profile_ind, profile), #dataSettings.denormalize(profiles_tensor[0,-1,profile_ind,:].detach().numpy(), profile),
                           c=colors[step+1],label=f'{shot}.{time}',linestyle='--')
            axes[i,0].errorbar(x,
                               predicted_mean[profile_ind],
                               #yerr=predicted_std[i],
                               c=colors[step+1])
    for i,actuator in enumerate(plotted_actuators):
        actuator_ind=saved_state['actuators'].index(actuator)
        axes[i,2].plot(actuator_times,
                       dataSettings.denormalize(actuators_tensor[0,-(lookahead+1):,actuator_ind].detach().numpy(), actuator),
                       c=colors[step+1],marker='o',markersize=4)
        axes[i,2].set_ylabel(label_map[actuator])
axes[-1,0].set_xlabel(r'$\rho$')
axes[-1,1].set_xlabel('time (ms)')
axes[-1,2].set_xlabel('time (ms)')
axes[0,0].legend()
for ax in np.ndarray.flatten(axes):
    ax.axhline(0,linestyle='--',c='k')
plt.show()
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
