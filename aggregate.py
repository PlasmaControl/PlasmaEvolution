# NEED TO ADD NORMALIZATION, right now can only do same signal like temperature only

import pickle
import numpy as np
import torch
from dataSettings import normalizations

# which_blend='blend'
# relevant_profiles=['zipfit_etempfit_rho','zipfit_itempfit_rho'] #['zipfit_edensfit_rho','zeff_rho','qpsi_EFIT01']

# with open('tmp_blend_info.pkl','rb') as f:
#     info=pickle.load(f)
# truth=torch.Tensor(info['truth'])
# ensemble_sims=torch.Tensor(info[which_blend]['data'])
# sim_names=info[which_blend]['names']
# profiles=info['profiles']

class Blender(torch.nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        # nmodels, nsamples, nprofiles, ntimes, nrho
        initial_state=torch.ones(input_shape[0],1,1,1,1,requires_grad=True)
        self.x=torch.nn.Parameter(initial_state)
    def forward(self, data):
        coefficients=torch.nn.Softmax(0)(self.x)
        return torch.sum(coefficients*data,0)

class BlenderProfiles(torch.nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        # nmodels, nsamples, nprofiles, ntimes, nrho
        initial_state=torch.ones(input_shape[0],1,input_shape[2],1,1,requires_grad=True)
        self.x=torch.nn.Parameter(initial_state)
    def forward(self, data):
        coefficients=torch.nn.Softmax(0)(self.x)
        return torch.sum(coefficients*data,0)

class BlenderProfilesTimes(torch.nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        # nmodels, nsamples, nprofiles, ntimes, nrho
        initial_state=torch.ones(input_shape[0],1,input_shape[2],input_shape[3],1,requires_grad=True)
        self.x=torch.nn.Parameter(initial_state)
    def forward(self, data):
        coefficients=torch.nn.Softmax(0)(self.x)
        return torch.sum(coefficients*data,0)
    
class BlenderProfilesTimesRho(torch.nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        # nmodels, nsamples, nprofiles, ntimes, nrho
        initial_state=torch.ones(input_shape[0],1,input_shape[2],input_shape[3],input_shape[4],requires_grad=True)
        self.x=torch.nn.Parameter(initial_state)
    def forward(self, data):
        coefficients=torch.nn.Softmax(0)(self.x)
        return torch.sum(coefficients*data,0)

class BlenderNonlinear(torch.nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        self.model_degree=2
        # nmodels, nsamples, nprofiles, ntimes, nrho
        input_shape=torch.tensor(input_shape)
        # reorder so that we more naturally can select which dimensions to consider
        # nmodels is a first degree, also including nprofiles would be second
        # also ntimes would be third, and also rho would be fourth
        # this is because nn.Linear considers all dimensions before final flattened portion
        # as simply propagating through
        # nsamples, nrho, ntimes, nprofiles, nmodels
        self.permuted_dims=[1,4,3,2,0]
        input_dim=1
        for i in range(1,self.model_degree+1):
            input_dim*=input_shape[self.permuted_dims[-i]]
        output_dim=int(input_dim/input_shape[0]) # always reduce down nmodels
        hidden_dim=5
        extra_layers=0
        self.mlp=torch.nn.Sequential()
        self.mlp.append(torch.nn.Linear(input_dim, hidden_dim))
        self.mlp.append(torch.nn.ReLU())
        for layer in range(extra_layers):
            self.mlp.append(torch.nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(torch.nn.ReLU())
        self.mlp.append(torch.nn.Linear(hidden_dim, output_dim))
    def forward(self, data):
        permuted_output_shape=[data.shape[i] for i in self.permuted_dims]
        permuted_output_shape[-1]=1 # reduce down nmodels
        inverse_permuted_dims=[int(i) for i in torch.argsort(torch.Tensor(self.permuted_dims))]
        reordered_data=torch.permute(data,self.permuted_dims)
        flattened_data=torch.flatten(reordered_data,start_dim=-self.model_degree)
        mlp_data=self.mlp(flattened_data)
        mlp_data_expanded=mlp_data.reshape(permuted_output_shape)
        mlp_data_unpermuted=mlp_data_expanded.permute(inverse_permuted_dims)
        return mlp_data_unpermuted.flatten(start_dim=0,end_dim=1) # we reduced down models

model_name_map={'Blender': Blender, 'BlenderProfiles': BlenderProfiles, 'BlenderProfilesTimes': BlenderProfilesTimes,
                'BlenderNonlinear': BlenderNonlinear}
    
def train_model(ensemble_sims,truth,
                profiles,relevant_profiles,
                model_filename,
                model_type='Blender',chunk_size=20):
    ensemble_sims=torch.Tensor(ensemble_sims)
    truth=torch.Tensor(truth)
    model_hyperparams={'input_shape': ensemble_sims.shape}
    model=model_name_map[model_type](**model_hyperparams)
    num_chunks=int(len(truth)/chunk_size)
    to_mask=torch.ones(truth.shape)
    for i in range(len(ensemble_sims)):
        to_mask[torch.isnan(ensemble_sims[i])]=0
        truth[torch.isnan(ensemble_sims[i])]=0
        ensemble_sims[i][torch.isnan(ensemble_sims[i])]=0
    # profile_inds_to_mask=[]
    # for profile in profiles:
    #     if profile not in relevant_profiles:
    #         profile_inds_to_mask.append(profiles.index(profile))
    # to_mask[:,profile_inds_to_mask,:,:]=0
    x_chunks=[ensemble_sims[:,i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    y_chunks=[truth[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    mask_chunks=[to_mask[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

    lr=1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn=torch.nn.MSELoss(reduction='sum')
    model.train()
    avg_train_losses=[]
    train_losses=[]
    for epoch in range(100):
        for i in range(len(x_chunks)):
            x=x_chunks[i]
            y=y_chunks[i]
            mask=mask_chunks[i]
            output=model(x)
            output=mask*output
            target=mask*y
            optimizer.zero_grad()
            train_loss=loss_fn(output, target) / torch.count_nonzero(mask)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
        avg_train_losses.append(sum(train_losses)/len(train_losses))
    print(avg_train_losses)
    #print(torch.squeeze(torch.nn.Softmax(0)(model.x)))
    torch.save({'model_state_dic': model.state_dict(),
                'model_hyperparams': model_hyperparams,
                'model_type': model_type},
               model_filename)

def inference_model(model_filename, ensemble_sims):
    ensemble_sims=torch.Tensor(ensemble_sims)
    saved_state=torch.load(model_filename, map_location=torch.device('cpu'))
    model=model_name_map[saved_state['model_type']](**saved_state['model_hyperparams'])
    model.load_state_dict(saved_state['model_state_dic'])
    yhat=model(ensemble_sims)
    return yhat
