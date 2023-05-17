import torch
torch.manual_seed(0)
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import time

import customModels
import customLosses
import customDatasetMakers
from dataSettings import nx, train_shots, val_shots

import configparser
import os
import sys

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename='configs/default.cfg'

config=configparser.ConfigParser()
config.read(config_filename)
data_filename=config['data']['data_filename']
model_type=config['model']['model_type']
n_epochs=int(config['optimization']['n_epochs'])
batch_size=int(config['optimization']['batch_size'])
lr=float(config['optimization']['lr'])
energyWeight=float(config['optimization']['energyWeight'])
lookahead=int(config['inputs']['lookahead'])
lookback=int(config['inputs']['lookback'])
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs']['parameters'].split()

# dump to same location as the config filename, with .tar instead of .cfg
output_filename=os.path.basename(config_filename).split('.cfg')[0]+".tar"

if model_type=='PlasmaGRU':
    model = customModels.PlasmaGRU(profiles, actuators, parameters)
    loss_fn = customLosses.combinedLoss(energyWeight)
else:
    model = customModels.PlasmaConv2D(profiles, actuators, parameters)
    loss_fn = customLosses.myMSELoss()

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
    for profiles_train, actuators_train, parameters_train, _ in train_loader:
        profiles_train=profiles_train.to(device)
        actuators_train=actuators_train.to(device)
        parameters_train=parameters_train.to(device)
        optimizer.zero_grad()
        output_profiles_hat=model(profiles_train, actuators_train, parameters_train)
        train_loss=loss_fn(output_profiles_hat,
                           profiles_train, actuators_train, parameters_train,
                           profiles, actuators, parameters)
        # Backpropagation
        train_loss.backward()
        optimizer.step()
        train_losses[-1]+=train_loss.item()*len(profiles_train) # mean * # samples in batch
    train_losses[-1]/=len(train_dataset) # now divide by total number of samples to get mean over steps/batches
    model.eval()
    with torch.no_grad():
        val_losses.append(0)
        for profiles_val, actuators_val, parameters_val, _ in val_loader:
            profiles_val=profiles_val.to(device)
            actuators_val=actuators_val.to(device)
            parameters_val=parameters_val.to(device)
            output_profiles_hat = model(profiles_val, actuators_val, parameters_val)
            val_loss = loss_fn(output_profiles_hat,
                               profiles_val, actuators_val, parameters_val,
                               profiles, actuators, parameters)
            val_losses[-1]+=val_loss.item()*len(profiles_val) # mean * # samples in batch
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
            'parameters': parameters,
            'lookahead': lookahead,
            'lookback': lookback,
            'exclude_ech': True
        }, output_filename)
    prev_time=time.time()

print(f'...took {(time.time()-start_time)/60:0.2f}min')
