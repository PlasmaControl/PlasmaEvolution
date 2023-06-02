import torch
import numpy as np
import customDatasetMakers
import customModels
from torch.utils.data import DataLoader
import dataSettings

import matplotlib.pyplot as plt

model_filenames=['default.tar'] #['lowIpEPEDOnlyOutliersExcluded.tar', 'lowIpSimpleOutliersExcluded.tar', 'lowIpEPEDOutliersExcluded.tar']
datasetTypes=['aug'] #['aug','lowIp', 'highIp']
colors={'lowIp': 'b', 'highIp': 'r', 'aug': 'g'}

model_labels={model_filename: model_filename for model_filename in model_filenames}
dataset_labels={datasetType: datasetType for datasetType in datasetTypes}

model_labels.update({'default.tar': 'test', 'lowIpSimpleOutliersExcluded.tar': 'data only', 'lowIpEPEDOnlyOutliersExcluded.tar': 'EPED only', 'lowIpEPEDOutliersExcluded.tar': 'both'})
dataset_labels.update({'lowIp': 'ip<1.3e6', 'highIp': 'ip>1.4e6', 'aug': 'aug'})

avg_dic={model_filename: {datasetType: None for datasetType in datasetTypes} for model_filename in model_filenames}
for file_ind,model_filename in enumerate(model_filenames):
    saved_state=torch.load(model_filename)
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
                   'rnn': False}
    for datasetType in datasetTypes:
        print(datasetType)
        data_filename='test.h5'
        shots=dataSettings.test_shots
        if datasetType=='lowIp':
            datasetParams['ip_maximum']=1.3e6
            datasetParams['ip_minimum']=None
        elif datasetType=='highIp':
            datasetParams['ip_minimum']=1.4e6
            datasetParams['ip_maximum']=None
        elif datasetType=='aug':
            data_filename='aug_data.h5'
            shots=None
        dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,saved_state['parameters'],
                                                     **datasetParams,shots=shots)
        model.eval()
        with torch.no_grad():
            profiles_test,actuators_test,extra_info=dataset[:]
            output_profiles_hat=model(profiles_test, actuators_test)
            avg_dic[model_filename][datasetType]=torch.mean(torch.nn.MSELoss(reduction='none')(output_profiles_hat,profiles_test),axis=0)

width=1./len(datasetTypes)
ylabels=['te[26]'] #,['te[6]','te[26]','ne[6]','ne[26]']
include_mean=False
fig,axes=plt.subplots(len(ylabels)+include_mean,sharex=True)
axes=np.atleast_1d(axes)
for file_ind,model_filename in enumerate(model_filenames):
    for datasetTypeInd,datasetType in enumerate(avg_dic[model_filename].keys()):
        for inputInd in range(len(ylabels)):
            if file_ind==0:
                label=dataset_labels[datasetType]
            else:
                label=None
            axes[inputInd].bar(datasetTypeInd*width+2*file_ind,avg_dic[model_filename][datasetType][inputInd],width,align='edge',
                    color=colors[datasetType],label=label)
            axes[inputInd].set_xticks(2*np.arange(len(model_filenames))+0.5,labels=[model_labels[model_filename] for model_filename in model_filenames])
            axes[inputInd].set_ylabel(ylabels[inputInd])
        if include_mean:
            inputInd=len(ylabels)
            axes[inputInd].bar(datasetTypeInd*width+2*file_ind,torch.mean(avg_dic[model_filename][datasetType]),width,align='edge',
                    color=colors[datasetType],label=label)
            axes[inputInd].set_xticks(2*np.arange(len(model_filenames))+0.5,labels=[model_labels[model_filename] for model_filename in model_filenames])
            axes[inputInd].set_ylabel('mean')
axes[0].legend()
plt.show()
