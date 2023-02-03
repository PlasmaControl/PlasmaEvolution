import torch
import numpy as np
import customDatasetMakers
import customModels
from torch.utils.data import DataLoader
from dataSettings import nx, normalizations, val_shots

import matplotlib.pyplot as plt
from matplotlib import cm
from torchviz import make_dot

input_filename='PlasmaGRU.tar'
data_filename='example_174042_165400.h5'

saved_state=torch.load(input_filename)
model=customModels.PlasmaGRU(saved_state['profiles'], saved_state['actuators'], saved_state['parameters'])
model.load_state_dict(saved_state['model_state_dict'])

extra_sigs=saved_state['extra_sigs']
# these must be appended to the end since the order of the extra_sigs matters for models
# taking in these parameters
if 'shots' not in extra_sigs:
    extra_sigs.append('shots')
if 'times' not in extra_sigs:
    extra_sigs.append('times')
dataset=customDatasetMakers.standard_dataset(data_filename,saved_state['profiles'],saved_state['actuators'],saved_state['parameters'],
                                             saved_state['lookahead'],saved_state['lookback'], shots=val_shots[-100::5],
                                             latest_output_only=saved_state['latest_output_only'],extra_sigs=extra_sigs)
data_loader=DataLoader(dataset, batch_size=50)
output_profiles, input_profiles, input_actuators, input_parameters, extra_sigs_tensor = next(iter(data_loader))
recorded_shots=extra_sigs_tensor[:,extra_sigs.index('shots')]
recorded_times=extra_sigs_tensor[:,extra_sigs.index('times')]
yhat_numpy=model(input_profiles, input_actuators, input_parameters).detach().numpy()
input_profiles_numpy=input_profiles.detach().numpy()
output_profiles_numpy=output_profiles.detach().numpy()
for i,profile in enumerate(saved_state['profiles']):
    yhat_numpy[:,i,:]=customDatasetMakers.denormalize(yhat_numpy[:,i,:], profile)
    input_profiles_numpy[:,i,:]=customDatasetMakers.denormalize(input_profiles_numpy[:,i,:], profile)
    output_profiles_numpy[:,:,i,:]=customDatasetMakers.denormalize(output_profiles_numpy[:,:,i,:], profile)
input_actuators_numpy=input_actuators.detach().numpy()
for i,actuator in enumerate(saved_state['actuators']):
    input_actuators_numpy[:,:,i]=customDatasetMakers.denormalize(input_actuators_numpy[:,:,i], actuator)

batch_ind=8
DT=25
present_time=recorded_times[batch_ind].detach().numpy()
times=np.arange(present_time-DT*saved_state['lookback'],
                present_time+DT*(saved_state['lookahead']+1),
                DT)
shot=recorded_shots[batch_ind].detach().numpy()
x=np.linspace(0,1,nx)
nrows=max(len(saved_state['profiles']),len(saved_state['actuators']))
fig,axes=plt.subplots(nrows=nrows,ncols=2,sharex='col')
axes=axes.T
time_inds=np.arange(0,saved_state['lookahead'])
colors=cm.jet(np.linspace(0,1,len(time_inds)))
for i,profile in enumerate(saved_state['profiles']):
    axes[0,i].plot(x,input_profiles_numpy[batch_ind,i,:],label='input',c='k')
    for time_count,time_ind in enumerate(time_inds): # loop over predicted timesteps
        output_label=None
        prediction_label=None
        if time_count==0:
            output_label='first step'
            prediction_label='prediction'
        axes[0,i].plot(x,output_profiles_numpy[batch_ind,time_ind,i,:],label=output_label,c=colors[time_count])
        axes[0,i].plot(x,yhat_numpy[batch_ind,time_ind,i,:],label=prediction_label,c=colors[time_count],linestyle='--')
    axes[0,i].set_ylabel(profile)
    axes[0,i].set_xlim(0,1)
    if 'qpsi' in profile:
        axes[0,i].set_ylim(0.5,5)
    if 'etempfit' in profile or 'edensfit' in profile or 'itempfit' in profile or 'pres_' in profile:
        axes[0,i].set_ylim(0,None)
axes[0,0].legend()
for i,actuator in enumerate(saved_state['actuators']):
    axes[1,i].plot(times,input_actuators_numpy[batch_ind,:,i])
    axes[1,i].set_ylabel(actuator)
    axes[1,i].axvline(present_time,c='k',linestyle='--')
fig.suptitle(f'{shot}.{present_time}')
fig.show()
plt.show()

