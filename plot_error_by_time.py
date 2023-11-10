import matplotlib.pyplot as plt

import numpy as np
import plotting_helpers
import pickle
from dataSettings import nx, DT
from plotting_helpers import label_map

min_step=0
max_step=7
pickle_filename='YesGasNoDen.pkl'
title=f"{pickle_filename.split('.pkl')[0]}boxplot.png"

num_steps=max_step-min_step
with open(pickle_filename,'rb') as f:
    data=pickle.load(f)
samples=list(data.keys())
num_samples=len(samples)

sample=samples[0]
nwarmup=len(data[sample]['truth']['times'])-len(data[sample]['predictions']['times'])
profiles=list(data[sample]['predictions']['profiles'].keys())
parameters=list(data[sample]['predictions']['parameters'].keys())
errors={}
for sig in profiles:
    #errors[sig]=np.zeros((num_samples,num_steps,nx))*np.nan
    errors[sig]=np.zeros((num_samples,num_steps))*np.nan
for sig in parameters:
    errors[sig]=np.zeros((num_samples,num_steps))*np.nan
for sample_ind, sample in enumerate(samples):
    sample_num_steps=len(data[sample]['predictions']['times'])
    for sig in profiles:
        truth=data[sample]['truth']['profiles'][sig][nwarmup+min_step:max_step]
        prediction=np.nanmean(data[sample]['predictions']['profiles'][sig],axis=0)[min_step:max_step]
        error=np.sqrt(np.nanmean(np.square(truth-prediction)/np.square(truth),axis=-1))
        errors[sig][sample_ind][:sample_num_steps]=error
    for sig in parameters:
        truth=data[sample]['truth']['parameters'][sig][nwarmup+min_step:max_step]
        prediction=np.nanmean(data[sample]['predictions']['parameters'][sig],axis=0)[min_step:max_step]
        error=np.sqrt(np.nanmean(np.square(truth-prediction)/np.square(truth)))
        errors[sig][sample_ind][:sample_num_steps]=error
fig,axes=plt.subplots(len(profiles+parameters),sharex=True)
axes=np.atleast_1d(axes)
for i,sig in enumerate(profiles+parameters):
    axes[i].boxplot(errors[sig])
    axes[i].set_ylabel(label_map[sig])
    axes[i].set_ylim((0,1))
axes[-1].set_xticks(range(min_step+1,max_step+1), [int((step+1)*DT*1e3) for step in range(min_step,max_step)])
axes[-1].set_xlabel('prediction length (ms)')
fig.supylabel(r'$\sigma$ (% error)')
plt.savefig(title)
plt.show()
