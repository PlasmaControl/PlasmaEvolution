import configparser
import torch
#torch.manual_seed(0)
import dataSettings
import numpy as np
import os
import glob
import re
from customModels import IanRNN, IanMLP, HiroLinear
from customDatasetMakers import state_to_dic, dic_to_state

models={'IanRNN': IanRNN, 'IanMLP': IanMLP, 'HiroLinear': HiroLinear}

def get_considered_models(config_filename, ensemble=True, epoch=None):
    config=configparser.ConfigParser()
    config.read(config_filename)
    output_filename_base=config['model']['output_filename_base']
    output_dir=config['model']['output_dir']
    model_type=config['model']['model_type']
    profiles=config['inputs']['profiles'].split()
    actuators=config['inputs']['actuators'].split()
    parameters=config['inputs']['parameters'].split()
    calculations=config['inputs']['calculations'].split()
    state_length=len(profiles)*dataSettings.nx+len(parameters)
    actuator_length=len(actuators)
    calculation_length=len(calculations)*dataSettings.nx
    considered_models=[]
    epoch_specification=''
    if epoch is not None:
        epoch_specification=f'EPOCH{epoch}'
    if ensemble:
        regex=f'{output_filename_base}[0-9]*{epoch_specification}.tar'
        all_model_files=glob.glob(os.path.join(output_dir, regex))
        # glob is not as powerful, the star is for everything - so whittle down further so it only
        # accepts repeats of numbers
        all_model_files=[model_file for model_file in all_model_files if re.match(regex,os.path.basename(model_file))]
        # exclude models under the median loss
        #losses = []
        #for model_file in all_model_files:
        #    saved_state=torch.load(model_file, map_location=torch.device('cpu'))
        #    losses.append(np.min([saved_state['val_losses'][-i] for i in range(10)]))
        #max_loss = np.median(losses)
        for model_file in all_model_files:
            saved_state=torch.load(model_file, map_location=torch.device('cpu'))
            if True: #np.min([saved_state['val_losses'][-i] for i in range(10)])<max_loss:
                model=models[model_type](input_dim=state_length+calculation_length+2*actuator_length, output_dim=state_length,
                                         **saved_state['model_hyperparams'])
                model.load_state_dict(saved_state['model_state_dict'])
                considered_models.append(model)
        print(f'{len(considered_models)} models used')
        #print(f'{len(considered_models)}/{len(all_model_files)} models used (i.e. only loss<{max_loss:0.2e})')
    else:
        model_file=os.path.join(output_dir, f'{output_filename_base}{epoch_specification}.tar')
        saved_state=torch.load(model_file, map_location=torch.device('cpu'))
        model=models[model_type](input_dim=state_length+calculation_length+2*actuator_length, output_dim=state_length,
                                 **saved_state['model_hyperparams'])
        model.load_state_dict(saved_state['model_state_dict'])
        considered_models=[model]
        print(f'Using {model_file}')
    return considered_models

def get_fake_actuator_state(normalized_true_state, profiles, parameters, actuators):
    fake_actuator_state=normalized_true_state.clone()
    fake_actuator_dic=state_to_dic(fake_actuator_state, profiles=profiles, parameters=parameters, actuators=actuators)
    for actuator in actuators:
        arr = fake_actuator_dic[actuator][0]
        freeze_index = len(arr)//2 - 40
        perturb_index = len(arr)//2 + 30
        perturb_length = 20
        arr[freeze_index:-1] = torch.tensor([arr[freeze_index]]*len(arr[freeze_index:-1]))
        if (actuator=='D_tot'):
            perturb_index = len(arr)//2 + 30
            perturb_length = 30
            arr[perturb_index:perturb_index + perturb_length] = torch.tensor([((np.sin(np.pi*i/perturb_length))*arr[perturb_index] + arr[perturb_index]) for i in range(perturb_length)])
    fake_actuator_state = dic_to_state(fake_actuator_dic, profiles, parameters, actuators=actuators)
    return fake_actuator_state
