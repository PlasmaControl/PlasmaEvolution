import torch
import numpy as np
import customDatasetMakers
import customModels
from torch.utils.data import DataLoader
import dataSettings

import matplotlib.pyplot as plt

shot=174754
input_filename='default.tar' #'default.tar'
data_filename='test.h5'

name_map={'zipfit_edensfit_rho': 'ne',
          'zipfit_etempfit_rho': 'te',
          'betan_EFIT01': 'betan',
          'rmaxis_EFIT01': 'r0',
          'aminor_EFIT01': 'a',
          'kappa_EFIT01': 'kappa',
          'tritop_EFIT01': 'tritop',
          'tribot_EFIT01': 'tribot',
          'epedHeightForNe1': 'eped1',
          'epedHeightForNe3': 'eped3',
          'epedHeightForNe5': 'eped5',
          'epedHeightForNe7': 'eped7'}

saved_state=torch.load(input_filename)

profiles=saved_state['profiles']
actuators=saved_state['actuators']
space_inds=saved_state['space_inds']

model=customModels.ProfilesFromActuators(profiles,
                                         actuators,
                                         len(space_inds))
model.load_state_dict(saved_state['model_state_dict'])
datasetParams={'lookahead': saved_state['lookahead'],
               'lookback': saved_state['lookback'],
               'space_inds': space_inds,
               'rnn': False,
               'ip_minimum': 1.4e6}
dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,saved_state['parameters'],
                                             **datasetParams,shots=[shot])
model.eval()

profiles_test, actuators_test, extra_info = dataset[:]
train_start=extra_info[0,0]
output_profiles_hat=model(profiles_test, actuators_test)
for i,profile in enumerate(profiles):
    profiles_test[:,i*len(space_inds):(i+1)*len(space_inds)] = dataSettings.denormalize(profiles_test[:,i*len(space_inds):(i+1)*len(space_inds)], profile)
    output_profiles_hat[:,i*len(space_inds):(i+1)*len(space_inds)] = dataSettings.denormalize(output_profiles_hat[:,i*len(space_inds):(i+1)*len(space_inds)], profile)
for i,actuator in enumerate(actuators):
    actuators_test[:,i] = dataSettings.denormalize(actuators_test[:,i], actuator)
with torch.no_grad():
    fig,axes=plt.subplots(len(profiles)*len(space_inds)+len(actuators),sharex=True)
    plot_ind=0
    for i in range(len(profiles)):
        for j in range(len(space_inds)):
            plot_ind=i*len(space_inds)+j
            axes[plot_ind].plot(extra_info[:,0]+extra_info[:,1]-train_start,profiles_test[:,plot_ind],label='truth')
            axes[plot_ind].plot(extra_info[:,0]+extra_info[:,1]-train_start,output_profiles_hat[:,plot_ind],label='hat')
            ylabel=profiles[i]
            if profiles[i] in name_map:
                ylabel=name_map[profiles[i]]
            axes[plot_ind].set_ylabel(f'{ylabel}[{space_inds[j]}]')
    axes[0].legend()
    for i in range(len(actuators)):
        plot_ind=len(profiles)*len(space_inds)+i
        axes[plot_ind].plot(extra_info[:,0]+extra_info[:,1]-train_start,actuators_test[:,i])
        ylabel=actuators[i]
        if actuators[i] in name_map:
            ylabel=name_map[actuators[i]]
        axes[plot_ind].set_ylabel(ylabel)
    fig.suptitle(shot)
    plt.show()
