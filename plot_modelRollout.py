import numpy as np
import plotting_helpers
import pickle

pickle_filename='YesGasNoDen.pkl'

rho_ind=10

with open(pickle_filename,'rb') as f:
    data=pickle.load(f)
key=list(data.keys())[0]

title=f"{key}_{pickle_filename.split('.pkl')[0]}"
predicted_means={}
predicted_stds={}
denormalized_true_dic={}
plotted_profiles=[sig for sig in data[key]['truth']['profiles']]
plotted_parameters=[sig for sig in data[key]['truth']['parameters']]
plotted_actuators=[sig for sig in data[key]['truth']['actuators']]
for sig_type in ['profiles','parameters']:
    for sig in data[key]['predictions'][sig_type]:
        predicted_means[sig]=np.mean(data[key]['predictions'][sig_type][sig],axis=0)
        predicted_stds[sig]=np.std(data[key]['predictions'][sig_type][sig],axis=0)
for sig_type in ['profiles','parameters','actuators']:
    for sig in data[key]['truth'][sig_type]:
        denormalized_true_dic[sig]=data[key]['truth'][sig_type][sig]
plotting_helpers.modelRollout_plot(predicted_means, predicted_stds, data[key]['predictions']['times'],
                                   denormalized_true_dic, data[key]['truth']['times'],
                                   plotted_profiles, plotted_parameters, plotted_actuators,
                                   rho_ind, title)
