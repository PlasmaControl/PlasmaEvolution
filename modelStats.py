import matplotlib.pyplot as plt
import sys
import os
import configparser
import torch
#torch.manual_seed(0)
import customModels
import customLosses
import customDatasetMakers
import glob

plot_ensemble=False

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename=f'configs/default.cfg'
config=configparser.ConfigParser()
config.read(config_filename)
output_filename_base=config['model']['output_filename_base']
output_dir=config['model']['output_dir']

if plot_ensemble:
    all_model_files=glob.glob(os.path.join(output_dir, f'{output_filename_base}*.tar'))
else:
    all_model_files=[os.path.join(output_dir, f'{output_filename_base}.tar')]

for model_ind, input_filename in enumerate(all_model_files):
    model=os.path.join(input_filename)
    saved_state=torch.load(model, map_location=torch.device('cpu'))
    plt.plot(saved_state['train_losses'],c='r',label='train')
    plt.plot(saved_state['val_losses'],c='b',label='validation')
    if plot_ensemble:
        plt.text(len(saved_state['val_losses'])-1, saved_state['val_losses'][-1], os.path.basename(input_filename))
    if model_ind==0:
        plt.legend()
plt.ylabel('loss')
plt.xlabel('epoch')
if not plot_ensemble:
    plt.title(f'{output_filename_base}')
plt.savefig(f'{output_filename_base}_stats.svg', format='svg')
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
