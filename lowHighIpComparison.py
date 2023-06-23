import torch
import numpy as np
import customDatasetMakers
import customModels
from torch.utils.data import DataLoader
import dataSettings

import matplotlib.pyplot as plt

model_filenames=['eped','lowIpTeEPED.tar','lowIpTeData.tar','lowIpTeEPEDAndData.tar']
default_filename='lowIpTeEPED.tar' # which filename to use for data settings for EPED
datasetTypes=['lowIp', 'highIp','aug']
colors={'lowIp': 'b', 'highIp': 'r', 'aug': 'g'}

plot_bar=True
plot_hist=True

model_labels={model_filename: model_filename for model_filename in model_filenames}
dataset_labels={datasetType: datasetType for datasetType in datasetTypes}

model_labels.update({'eped': 'EPED-NN',
#                     'lowIpData.tar': 'NN (data)', 'lowIpEPED.tar': 'NN (EPED)', 'lowIpEPEDAndData.tar': 'NN (data+EPED)',
                     'lowIpTeData.tar': 'NN (data)', 'lowIpTeEPED.tar': 'NN (EPED-NN)', 'lowIpTeEPEDAndData.tar': 'NN (data+EPED-NN)'})
dataset_labels.update({'lowIp': r'Low $I_p$ ($<1.3MA$)', 'highIp': r'High $I_p$ ($>1.4MA$)', 'aug': 'AUG'})

mse_dic={model_filename: {datasetType: None for datasetType in datasetTypes} for model_filename in model_filenames}
offset_dic={model_filename: {datasetType: None for datasetType in datasetTypes} for model_filename in model_filenames}
for file_ind,model_filename in enumerate(model_filenames):
    if model_filename=='eped':
        filename=default_filename
    else:
        filename=model_filename
    saved_state=torch.load(filename)
    profiles=saved_state['profiles']
    actuators=saved_state['actuators']
    space_inds=saved_state['space_inds']
    datasetParams={'lookahead': saved_state['lookahead'],
                   'lookback': saved_state['lookback'],
                   'space_inds': space_inds,
                   'rnn': False}
    if model_filename=='eped':
        actuators.append('eped_te_prediction')
    else:
        model=customModels.ProfilesFromActuators(profiles,
                                                 actuators,
                                                 len(space_inds))
        model.load_state_dict(saved_state['model_state_dict'])
    for datasetType in datasetTypes:
        if datasetType=='lowIp':
            data_filename='test.h5'
            shots=dataSettings.test_shots[:400]
            datasetParams['ip_minimum']=None
            datasetParams['ip_maximum']=1.3e6
        elif datasetType=='highIp':
            data_filename='test.h5'
            shots=dataSettings.test_shots[:200]
            datasetParams['ip_minimum']=1.4e6
            datasetParams['ip_maximum']=None
        elif datasetType=='aug':
            data_filename='aug_data.h5'
            shots=None
            datasetParams['ip_minimum']=None
            datasetParams['ip_maximum']=None
        dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,saved_state['parameters'],
                                                     **datasetParams,shots=shots)
        if model_filename=='eped':
            profiles_test,actuators_test,extra_info=dataset[:]
            output_profiles_hat=actuators_test[:,-1]
            import pdb; pdb.set_trace()
        else:
            model.eval()
            with torch.no_grad():
                profiles_test,actuators_test,extra_info=dataset[:]
                output_profiles_hat=model(profiles_test, actuators_test)
                #mse_dic[model_filename][datasetType]=torch.nn.MSELoss(reduction='none')(output_profiles_hat,profiles_test)
        mse_dic[model_filename][datasetType]=100*(output_profiles_hat-profiles_test)**2 #/profiles_test**2
        offset_dic[model_filename][datasetType]=100*(output_profiles_hat-profiles_test) #/profiles_test
width=1./len(datasetTypes)
hist_lims=[-100,100]
ylabels=['MSE Error (Edge Te prediction)'] #,['te[6]','te[26]','ne[6]','ne[26]']
if plot_bar:
    fig_bar,axes_bar=plt.subplots(len(ylabels),sharex=True,sharey=True)
    axes_bar=np.atleast_1d(axes_bar)
if plot_hist:
    fig_hist,axes_hist=plt.subplots(len(model_filenames),len(datasetTypes),sharex=True,sharey=True)
for file_ind,model_filename in enumerate(model_filenames):
    for datasetTypeInd,datasetType in enumerate(datasetTypes):
        for inputInd in range(len(ylabels)):
            if file_ind==0:
                label=dataset_labels[datasetType]
            else:
                label=None
            if plot_bar:
                axes_bar[inputInd].bar(datasetTypeInd*width+2*file_ind,torch.mean(mse_dic[model_filename][datasetType],axis=0)[inputInd],width,align='edge',
                                       color=colors[datasetType],label=label)
                axes_bar[inputInd].set_xticks(2*np.arange(len(model_filenames))+0.5,labels=[model_labels[model_filename] for model_filename in model_filenames])
                axes_bar[inputInd].set_ylabel(ylabels[inputInd])
            if plot_hist:
                bins=np.linspace(hist_lims[0],hist_lims[1],20)
                axes_hist[file_ind,datasetTypeInd].hist(offset_dic[model_filename][datasetType][:,inputInd],bins=bins,color=colors[datasetType])
                axes_hist[file_ind,datasetTypeInd].set_xlim(hist_lims)
        # if include_mean:
        #     inputInd=len(ylabels)
        #     if plot_bar:
        #         axes[inputInd].bar(datasetTypeInd*width+2*file_ind,torch.mean(avg_dic[model_filename][datasetType]),width,align='edge',
        #                 color=colors[datasetType],label=label)
        #         axes[inputInd].set_xticks(2*np.arange(len(model_filenames))+0.5,labels=[model_labels[model_filename] for model_filename in model_filenames])
        #     axes[inputInd].set_ylabel('mean')
if plot_bar:
    axes_bar[0].legend()
if plot_hist:
    for datasetTypeInd,datasetType in enumerate(datasetTypes):
        axes_hist[0,datasetTypeInd].set_title(dataset_labels[datasetType])
    for file_ind,model_filename in enumerate(model_filenames):
        axes_hist[file_ind,0].set_ylabel(model_labels[model_filename])
    axes_hist[-1,int(len(datasetTypes)/2)].set_xlabel('Offset (Edge Te prediction)')
plt.show()
