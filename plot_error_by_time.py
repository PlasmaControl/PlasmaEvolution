import matplotlib.pyplot as plt

import numpy as np
import plotting_helpers
import pickle
from dataSettings import nx, DT
from plotting_helpers import label_map

min_step=0
max_step=20
#pickle_filename='rollout_allECHwithSim.pkl'
#title=f"{pickle_filename.split('.pkl')[0]}boxplot.png"

title="noECH_time_error_comparison.png"

for file_index,pickle_filename in enumerate(['rollout_noECHwithSim.pkl', 'rollout_noECHnoSim.pkl']):
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
        # max_index is a workaround to ensure we don't write out of bounds
        # for samples that have fewer timesteps than min_step
        max_index=sample_num_steps-min_step
        if sample_num_steps>=min_step:
            for sig in profiles:
                truth=data[sample]['truth']['profiles'][sig][min_step+nwarmup:max_step+nwarmup]
                prediction=np.nanmean(data[sample]['predictions']['profiles'][sig],axis=0)[min_step:max_step]
                error=np.sqrt(np.nanmean(np.square(truth-prediction)/np.square(truth),axis=-1))
                errors[sig][sample_ind][:max_index]=error
            for sig in parameters:
                truth=data[sample]['truth']['parameters'][sig][nwarmup+min_step:max_step]
                prediction=np.nanmean(data[sample]['predictions']['parameters'][sig],axis=0)[min_step:max_step]
                error=np.sqrt(np.nanmean(np.square(truth-prediction)/np.square(truth)))
                errors[sig][sample_ind][:max_index]=error
    if file_index==0:
        fig,axes=plt.subplots(len(profiles+parameters),sharex=True)
    axes=np.atleast_1d(axes)
    for i,sig in enumerate(profiles+parameters):
        errors_for_plot=[]
        mean_error=[]
        std_error=[]
        for step_ind in range(num_steps):
            error=errors[sig][:,step_ind]
            error=error[~np.isnan(error)]
            errors_for_plot.append(error)
            mean_error.append(np.median(error))
            std_error.append(np.subtract(*np.percentile(error, [75, 25])))
        #axes[i].boxplot(errors_for_plot)
        axes[i].errorbar([int((min_step+step_ind+1)*DT*1e3) for step_ind in range(num_steps)],
                         mean_error, std_error,
                         label=pickle_filename)
        #axes[i].plot([int((min_step+step_ind+1)*DT*1e3) for step_ind in range(num_steps)], mean_error)
        axes[i].set_ylabel(label_map[sig])
        axes[i].set_ylim((0,0.5))
#axes[-1].set_xticks([step_ind+1 for step_ind in range(num_steps)],[int((min_step+step_ind+1)*DT*1e3) for step_ind in range(num_steps)])
axes[-1].set_xlabel('prediction length (ms)')
axes[0].legend()
fig.supylabel(r'$\sigma$ (% error)')
plt.savefig(title)
plt.show()
