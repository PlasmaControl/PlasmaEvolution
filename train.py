import torch
torch.manual_seed(0)
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import time

import customModels
import customDatasetMakers
from dataSettings import nx, train_shots, val_shots

data_filename='example_174042_165400.h5'

output_filename='PlasmaConv2D.tar'
profiles=['zipfit_etempfit_psi','zipfit_trotfit_psi','zipfit_edensfit_psi','pres_EFIT01','qpsi_EFIT01']
actuators=['pinj', 'tinj', 'ipsiptargt','dstdenp']
parameters=['li_EFIT01','tribot_EFIT01','tritop_EFIT01','dssdenest','kappa_EFIT01','volume_EFIT01']

lookahead=6
lookback=5

n_epochs=20
batch_size=10
lr=1e-2

if (train_shots is None) or (val_shots is None):
    dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,lookahead,lookback)
    ntrain=int(0.7*len(dataset))
    nval=int(0.2*len(dataset))
    ntest=len(dataset)-ntrain-nval
    train_dataset, val_dataset, _ = random_split(dataset,[ntrain,nval,ntest])
else:
    train_dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,lookahead,lookback,train_shots)
    val_dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,lookahead,lookback,val_shots)
train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=batch_size)

#model = customModels.ProfilesFromActuators(profiles, actuators)
model = customModels.PlasmaConv2D(profiles, actuators, parameters)
loss_fn = torch.nn.MSELoss(reduction='mean')

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
    if epoch and (not (epoch % 5)):
        print(f'{epoch:4d}/{n_epochs}({(time.time()-prev_time)/60:0.2f}min)... train: {train_losses[-1]:0.2e}, val: {val_losses[-1]:0.2e};')
        prev_time=time.time()
    model.train()
    train_losses.append(0)
    for output_profiles_train, input_profiles_train, input_actuators_train, input_parameters_train in train_loader:
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
        for output_profiles_val, input_profiles_val, input_actuators_val, input_parameters_val in val_loader:
            input_profiles_val=input_profiles_val.to(device)
            input_actuators_val=input_actuators_val.to(device)
            input_parameters_val=input_parameters_val.to(device)
            output_profiles_val=output_profiles_val.to(device)
            output_profiles_hat = model(input_profiles_val, input_actuators_val, input_parameters_val)
            val_loss = loss_fn(output_profiles_hat, output_profiles_val)
            val_losses[-1]+=val_loss.item()*len(output_profiles_val) # mean * # samples in batch
        val_losses[-1]/=len(val_dataset)
    if val_losses[-1]==min(val_losses):
        print(f"checkpoint: train_loss: {train_losses[-1]:0.2e}, val_loss: {val_losses[-1]:0.2e}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'profiles': profiles,
            'actuators': actuators,
            'parameters': parameters,
            'lookahead': lookahead,
            'lookback': lookback,
            'exclude_ech': True
        }, output_filename)

print(f'...took {(time.time()-start_time)/60:0.2f}min: train: {train_losses[-1]:0.2e}, val: {val_losses[-1]:0.2e}')
