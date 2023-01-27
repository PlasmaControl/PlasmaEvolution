import torch
import numpy as np
import customDatasetMakers
import customModels
from torch.utils.data import DataLoader
from dataSettings import nx, normalizations, val_shots

import matplotlib.pyplot as plt
from torchviz import make_dot

input_filename='PlasmaConv2D.tar'
data_filename='example_174042_165400.h5'

saved_state=torch.load(input_filename)
model=customModels.PlasmaConv2D(saved_state['profiles'], saved_state['actuators'], saved_state['parameters'])
model.load_state_dict(saved_state['model_state_dict'])

dataset=customDatasetMakers.standard_dataset(data_filename,saved_state['profiles'],saved_state['actuators'],saved_state['parameters'],
                                             saved_state['lookahead'],saved_state['lookback'], shots=val_shots[-100::5])
data_loader=DataLoader(dataset, batch_size=50)
output_profiles, input_profiles, input_actuators, input_parameters, recorded_shots, recorded_times = next(iter(data_loader))
yhat_numpy=model(input_profiles, input_actuators, input_parameters).detach().numpy()
input_profiles_numpy=input_profiles.detach().numpy()
output_profiles_numpy=output_profiles.detach().numpy()
for i,profile in enumerate(saved_state['profiles']):
    yhat_numpy[:,i,:]=customDatasetMakers.denormalize(yhat_numpy[:,i,:], profile)
    input_profiles_numpy[:,i,:]=customDatasetMakers.denormalize(input_profiles_numpy[:,i,:], profile)
    output_profiles_numpy[:,i,:]=customDatasetMakers.denormalize(output_profiles_numpy[:,i,:], profile)
input_actuators_numpy=input_actuators.detach().numpy()
for i,actuator in enumerate(saved_state['actuators']):
    input_actuators_numpy[:,:,i]=customDatasetMakers.denormalize(input_actuators_numpy[:,:,i], actuator)

highest_batch_ind=49
x=np.linspace(0,1,nx)
nrows=max(len(saved_state['profiles']),len(saved_state['actuators']))
for plot_count in range(10):
    batch_ind=highest_batch_ind-plot_count*4
    time=recorded_times[batch_ind].detach().numpy()
    DT=25
    times=np.arange(time-DT*saved_state['lookback'],time+DT*saved_state['lookahead'],DT)
    shot=recorded_shots[batch_ind].detach().numpy()
    fig,axes=plt.subplots(nrows=nrows,ncols=2,sharex='col')
    axes=axes.T
    for i,profile in enumerate(saved_state['profiles']):
        axes[0,i].plot(x,input_profiles_numpy[batch_ind,i,:],label='input',c='r')
        axes[0,i].plot(x,output_profiles_numpy[batch_ind,i,:],label='output',c='k')
        axes[0,i].plot(x,yhat_numpy[batch_ind,i,:],label='prediction',c='k',linestyle='--')
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
        axes[1,i].axvline(time,c='k',linestyle='--')
    fig.suptitle(f'{shot}.{time}')
    fig.show()
plt.show()

if False:
    make_dot(yhat).render("model",format="png")
    output_profiles_hat_list=[]
    output_profiles_list=[]
    model.eval()
    with torch.no_grad():
        for output_profiles, input_profiles, input_actuators, input_parameters in data_loader:
            output_profiles_list.append(output_profiles)
            output_profiles_hat_list.append(model(input_profiles, input_actuators, input_parameters))
