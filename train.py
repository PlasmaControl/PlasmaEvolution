import torch
#torch.manual_seed(0)
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import time

import customModels
import customLosses
import customDatasetMakers
from dataSettings import nx, train_shots, val_shots, test_shots, val_indices

import configparser
import os
import sys

import dataSettings

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename='configs/default.cfg'

config=configparser.ConfigParser()
config.read(config_filename)
data_filename=config['data']['data_filename']
use_preprocessed_data=(config['data']['use_preprocessed_data']=='True')
preprocessed_data_filenamebase=config['data']['preprocessed_data_filenamebase']
dump_preprocessed_data=(config['data']['dump_preprocessed_data']=='True')
ip_minimum=float(config['data']['ip_minimum'])
ip_maximum=float(config['data']['ip_maximum'])
model_type=config['model']['model_type']
n_epochs=int(config['optimization']['n_epochs'])
batch_size=int(config['optimization']['batch_size'])
lr=float(config['optimization']['lr'])
lr_gamma=float(config['optimization']['lr_gamma'])
energyWeight=float(config['optimization']['energyWeight'])
lookahead=int(config['inputs']['lookahead'])
lookback=int(config['inputs']['lookback'])
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs']['parameters'].split()
space_inds=[int(key) for key in config['inputs']['space_inds'].split()]
# if not defined, use all data points
if len(space_inds)==0:
    space_inds=list(range(dataSettings.nx))

# dump to same location as the config filename, with .tar instead of .cfg
output_filename=os.path.join(config['model']['output_dir'],config['model']['output_filename_base']+".tar")

datasetParams={'lookahead': lookahead, 'lookback': lookback,
               'space_inds': space_inds, 'ip_minimum': ip_minimum, 'ip_maximum': ip_maximum}
if model_type=='PlasmaGRU':
    model = customModels.PlasmaGRU(profiles, actuators, parameters)
    loss_fn = customLosses.combinedLoss(energyWeight)
elif model_type=='PlasmaConv2D':
    model = customModels.PlasmaConv2D(profiles, actuators, parameters)
    loss_fn = customLosses.myMSELoss()
else:
    datasetParams.update({'rnn': False})
    HIDDEN_SIZE=40
    model = customModels.ProfilesFromActuators(profiles, actuators, len(space_inds))
    loss_fn = customLosses.simpleMSELoss()

if use_preprocessed_data:
    train_dataset=torch.load(f'{preprocessed_data_filenamebase}train.pt')
    val_dataset=torch.load(f'{preprocessed_data_filenamebase}val.pt')
else:
    if (train_shots is None) or (val_shots is None):
        dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,**datasetParams)
        ntrain=int(0.8*len(dataset))
        nval=int(0.1*len(dataset))
        ntest=len(dataset)-ntrain-nval
        train_dataset, val_dataset, test_dataset = random_split(dataset,[ntrain,nval,ntest])
    else:
        train_dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,shots=train_shots,**datasetParams)
        val_dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,shots=val_shots,**datasetParams)
        if dump_preprocessed_data:
            test_dataset=customDatasetMakers.standard_dataset(data_filename,profiles,actuators,parameters,shots=test_shots,**datasetParams)
    if dump_preprocessed_data:
        torch.save(train_dataset,f'{preprocessed_data_filenamebase}train.pt')
        torch.save(val_dataset,f'{preprocessed_data_filenamebase}val.pt')
        torch.save(test_dataset,f'{preprocessed_data_filenamebase}test.pt')
train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=batch_size)

train_losses=[]
val_losses=[]
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50,70], gamma=lr_gamma, verbose=True)

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
    for *model_inputs, _ in train_loader:
        for i in range(len(model_inputs)):
            model_inputs[i]=model_inputs[i].to(device)
        optimizer.zero_grad()
        model_output=model(*model_inputs)
        train_loss=loss_fn(model_output,
                           *model_inputs,
                           profiles, actuators, parameters)
        # Backpropagation
        train_loss.backward()
        optimizer.step()
        train_losses[-1]+=train_loss.item()*len(model_inputs[0]) # mean * # samples in batch
    scheduler.step()
    train_losses[-1]/=len(train_dataset) # now divide by total number of samples to get mean over steps/batches
    model.eval()
    with torch.no_grad():
        val_losses.append(0)
        for *model_inputs, _ in val_loader:
            for i in range(len(model_inputs)):
                model_inputs[i]=model_inputs[i].to(device)
            model_output = model(*model_inputs)
            val_loss = loss_fn(model_output,
                               *model_inputs,
                               profiles, actuators, parameters)
            val_losses[-1]+=val_loss.item()*len(model_inputs[0]) # mean * # samples in batch
        val_losses[-1]/=len(val_dataset)
    print(f'{epoch+1:4d}/{n_epochs}({(time.time()-prev_time):0.2f}s)... train: {train_losses[-1]:0.2e}, val: {val_losses[-1]:0.2e};')
    if val_losses[-1]==min(val_losses):
        print(f"Checkpoint")
        torch.save({
            'epoch': epoch,
            'val_indices': val_indices,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'profiles': profiles,
            'actuators': actuators,
            'parameters': parameters,
            'lookahead': lookahead,
            'lookback': lookback,
            'space_inds': space_inds,
            'exclude_ech': True
        }, output_filename)
    prev_time=time.time()

print(f'...took {(time.time()-start_time)/60:0.2f}min')
