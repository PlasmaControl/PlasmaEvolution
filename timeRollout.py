import torch
import numpy as np
import customDatasetMakers
import customModels
from torch.utils.data import DataLoader
import dataSettings
import customLosses

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
                                             saved_state['lookahead'],saved_state['lookback'], shots=dataSettings.val_shots[-100::5],
                                             extra_sigs=extra_sigs)
data_loader=DataLoader(dataset, batch_size=50)
profiles_tensor, actuators_tensor, parameters_tensor, extra_sigs_tensor = next(iter(data_loader))
recorded_shots=extra_sigs_tensor[:,extra_sigs.index('shots')]
recorded_times=extra_sigs_tensor[:,extra_sigs.index('times')]
yhat_tensor=model(profiles_tensor, actuators_tensor, parameters_tensor).detach()
yhat_numpy=yhat_tensor.numpy()
profiles_numpy=profiles_tensor.detach().numpy()
for i,profile in enumerate(saved_state['profiles']):
    yhat_numpy[:,:,i,:]=dataSettings.denormalize(yhat_numpy[:,:,i,:], profile)
    profiles_numpy[:,:,i,:]=dataSettings.denormalize(profiles_numpy[:,:,i,:], profile)
actuators_numpy=actuators_tensor.detach().numpy()
for i,actuator in enumerate(saved_state['actuators']):
    actuators_numpy[:,:,i]=dataSettings.denormalize(actuators_numpy[:,:,i], actuator)

batch_ind=8
present_time=recorded_times[batch_ind].detach().numpy()
DT_milliseconds=int(dataSettings.DT*1e3)
times=np.arange(present_time-DT_milliseconds*saved_state['lookback'],
                present_time+DT_milliseconds*(saved_state['lookahead']+1),
                DT_milliseconds)
shot=recorded_shots[batch_ind].detach().numpy()
x=np.linspace(0,1,dataSettings.nx)
nrows=max(len(saved_state['profiles']),len(saved_state['actuators'])+1)
fig,axes=plt.subplots(nrows=nrows,ncols=2,sharex='col')
axes=axes.T
time_inds=np.arange(0,saved_state['lookahead'])
colors=cm.jet(np.linspace(0,1,len(time_inds)))
for i,profile in enumerate(saved_state['profiles']):
    axes[0,i].plot(x,profiles_numpy[batch_ind,-saved_state['lookahead']-1,i,:],label='input',c='k')
    for time_count,time_ind in enumerate(time_inds): # loop over predicted timesteps
        output_label=None
        prediction_label=None
        if time_count==0:
            output_label='first step'
            prediction_label='prediction'
        axes[0,i].plot(x,profiles_numpy[batch_ind,-saved_state['lookahead']+time_ind,i,:],label=output_label,c=colors[time_count])
        axes[0,i].plot(x,yhat_numpy[batch_ind,time_ind,i,:],label=prediction_label,c=colors[time_count],linestyle='--')
    axes[0,i].set_ylabel(profile)
    axes[0,i].set_xlim(0,1)
    if 'qpsi' in profile:
        axes[0,i].set_ylim(0.5,5)
    if 'etempfit' in profile or 'edensfit' in profile or 'itempfit' in profile or 'pres_' in profile:
        axes[0,i].set_ylim(0,None)
axes[0,0].legend()
for i,actuator in enumerate(saved_state['actuators']):
    axes[1,i].plot(times,actuators_numpy[batch_ind,:,i])
    axes[1,i].set_ylabel(actuator)
    axes[1,i].axvline(present_time,c='k',linestyle='--')

W_predicted, W_real=[], []
etemp_ind=saved_state['profiles'].index('zipfit_etempfit_rho')
itemp_ind=saved_state['profiles'].index('zipfit_itempfit_rho')
edens_ind=saved_state['profiles'].index('zipfit_edensfit_rho')
volume_ind=saved_state['parameters'].index('volume_EFIT01')
pinj_ind=saved_state['actuators'].index('pinj')
for time_ind in range(profiles_tensor.shape[1]):
    W_real.append(customLosses.calculate_W(profiles_tensor[batch_ind:batch_ind+1,time_ind,etemp_ind,:],
                                           profiles_tensor[batch_ind:batch_ind+1,time_ind,itemp_ind,:],
                                           profiles_tensor[batch_ind:batch_ind+1,time_ind,edens_ind,:],
                                           parameters_tensor[batch_ind:batch_ind+1,-1,volume_ind])[0])
for time_ind in range(yhat_tensor.shape[1]):
    W_predicted.append(customLosses.calculate_W(yhat_tensor[batch_ind:batch_ind+1,time_ind,etemp_ind,:],
                                                yhat_tensor[batch_ind:batch_ind+1,time_ind,itemp_ind,:],
                                                yhat_tensor[batch_ind:batch_ind+1,time_ind,edens_ind,:],
                                                parameters_tensor[batch_ind:batch_ind+1,-1,volume_ind])[0])
P_rollout=1e3*actuators_tensor[batch_ind,-saved_state['lookahead']-1:,pinj_ind] + dataSettings.ohmicPower
P_now=P_rollout[0]
W_now=W_real[-saved_state['lookahead']-1]
dWdt=(W_real[-saved_state['lookahead']]-W_real[-saved_state['lookahead']-1])/dataSettings.DT
taue_now=customLosses.calculate_taue(W_now,dWdt,P_now)
W_expected=[W_now]
for time_ind in range(saved_state['lookahead']):
    W_expected.append(W_expected[-1]+(-W_expected[-1]/taue_now + P_rollout[time_ind])*dataSettings.DT)
axes[1,-1].plot(times[-saved_state['lookahead']-2:],np.array(W_real)/1.e6,label='real')
axes[1,-1].plot(times[-saved_state['lookahead']:],np.array(W_predicted)/1.e6,label='predicted')
axes[1,-1].plot(times[-saved_state['lookahead']-1:],np.array(W_expected)/1.e6,label='expected')
axes[1,-1].set_ylabel('Wmhd (MJ)')
axes[1,-1].legend()
fig.suptitle(f'{shot}.{present_time}')
plt.show()
