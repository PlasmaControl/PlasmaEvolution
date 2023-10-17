import matplotlib.pyplot as plt
import sys
import os
import configparser
import torch
#torch.manual_seed(0)
from torch.utils.data import DataLoader
import customModels
import customLosses
import customDatasetMakers
import glob

if (len(sys.argv)-1) > 0:
    model_name=sys.argv[1]
else:
    model_name='IanRNN0'
output_dir='/projects/EKOLEMEN/profile_predictor/joe_hiro_models/'
model=os.path.join(output_dir,f'{model_name}.tar')
saved_state=torch.load(model, map_location=torch.device('cpu'))
plt.plot(saved_state['train_losses'],c='r',label='train')
plt.plot(saved_state['val_losses'],c='b',label='validation')
#plt.text(len(saved_state['val_losses']), saved_state['val_losses'][-1], str(i))
plt.legend()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title(f'{model_name}')
plt.savefig(f'{model_name}_stats.svg', format='svg')
plt.show()

#model=customModels.PlasmaConv2D(saved_state['profiles'], saved_state['actuators'], saved_state['parameters'])
#model.load_state_dict(saved_state['model_state_dict'])

'''
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

train_dataset=torch.load(f'{preprocessed_data_filenamebase}train.pt')
val_dataset=torch.load(f'{preprocessed_data_filenamebase}val.pt')

train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader=DataLoader(val_dataset, batch_size=batch_size)



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
'''
