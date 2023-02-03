import torch
torch.manual_seed(0)
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import time

import customModels
import customDatasetMakers
from dataSettings import nx, train_shots, val_shots
val_shots=None

data_filename='test.h5' #'example_174042_165400.h5'

profiles=['zipfit_etempfit_psi','zipfit_trotfit_psi','zipfit_edensfit_psi','pres_EFIT01','qpsi_EFIT01']
actuators=['pinj', 'tinj', 'ipsiptargt','dstdenp']
parameters=['li_EFIT01','tribot_EFIT01','tritop_EFIT01','dssdenest','kappa_EFIT01','volume_EFIT01']
# choose from "time", "shot", "taue"
extra_sigs=['shots', 'times']

#model = customModels.ProfilesFromActuators(profiles, actuators)
if False:
    output_filename='PlasmaConv2D.tar'
    latest_output_only=True
    model = customModels.PlasmaConv2D(profiles, actuators, parameters)
    loss_fn = torch.nn.MSELoss(reduction='mean')
else:
    output_filename='PlasmaGRU.tar'
    latest_output_only=False
    model = customModels.PlasmaGRU(profiles, actuators, parameters)
    loss_fn = torch.nn.MSELoss(reduction='mean')

lookahead=6
lookback=8

n_epochs=20
batch_size=10
lr=1e-2

if (train_shots is None) or (val_shots is None):
    dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,lookahead,lookback,
                                                 latest_output_only=latest_output_only, extra_sigs=extra_sigs)
    ntrain=int(0.7*len(dataset))
    nval=int(0.2*len(dataset))
    ntest=len(dataset)-ntrain-nval
    train_dataset, val_dataset, _ = random_split(dataset,[ntrain,nval,ntest])
else:
    train_dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,lookahead,lookback,train_shots,
                                                       latest_output_only=latest_output_only, extra_sigs=extra_sigs)
    val_dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,lookahead,lookback,val_shots,
                                                     latest_output_only=latest_output_only, extra_sigs=extra_sigs)
train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=batch_size)

train_losses=[]
val_losses=[]
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

print('Training...')
if torch.cuda.is_available():
    device='cuda'
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPU(s)")
else:
    device = 'cpu'
    print("Using CPU")
model.to(device)
start_time=time.time()
prev_time=start_time
for epoch in range(n_epochs):
    model.train()
    train_losses.append(0)
    for output_profiles_train, input_profiles_train, input_actuators_train, input_parameters_train, _ in train_loader:
        input_profiles_train=input_profiles_train.to(device)
        input_actuators_train=input_actuators_train.to(device)
        input_parameters_train=input_parameters_train.to(device)
        output_profiles_train=output_profiles_train.to(device)
        optimizer.zero_grad()
        output_profiles_hat=model(input_profiles_train, input_actuators_train, input_parameters_train)
        train_loss=loss_fn(output_profiles_hat, output_profiles_train)
        # Backpropagation
        train_loss.backward()
        optimizer.step()
        train_losses[-1]+=train_loss.item()*len(output_profiles_train) # mean * # samples in batch
    train_losses[-1]/=len(train_dataset) # now divide by total number of samples to get mean over steps/batches
    model.eval()
    with torch.no_grad():
        val_losses.append(0)
        for output_profiles_val, input_profiles_val, input_actuators_val, input_parameters_val, _ in val_loader:
            input_profiles_val=input_profiles_val.to(device)
            input_actuators_val=input_actuators_val.to(device)
            input_parameters_val=input_parameters_val.to(device)
            output_profiles_val=output_profiles_val.to(device)
            output_profiles_hat = model(input_profiles_val, input_actuators_val, input_parameters_val)
            val_loss = loss_fn(output_profiles_hat, output_profiles_val)
            val_losses[-1]+=val_loss.item()*len(output_profiles_val) # mean * # samples in batch
        val_losses[-1]/=len(val_dataset)
    print(f'{epoch+1:4d}/{n_epochs}({(time.time()-prev_time):0.2f}s)... train: {train_losses[-1]:0.2e}, val: {val_losses[-1]:0.2e};')
    if val_losses[-1]==min(val_losses):
        print(f"Checkpoint")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'profiles': profiles,
            'actuators': actuators,
            'extra_sigs': extra_sigs,
            'parameters': parameters,
            'lookahead': lookahead,
            'lookback': lookback,
            'latest_output_only': latest_output_only,
            'exclude_ech': True
        }, output_filename)
    prev_time=time.time()

print(f'...took {(time.time()-start_time)/60:0.2f}min')
