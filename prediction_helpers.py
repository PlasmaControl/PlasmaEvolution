import configparser
import torch
#torch.manual_seed(0)
import dataSettings
import numpy as np
import os
import glob
from customModels import IanRNN, IanMLP, HiroLinear

models={'IanRNN': IanRNN, 'IanMLP': IanMLP, 'HiroLinear': HiroLinear}

@torch.no_grad()
class ModelStepper:
    def __init__(self, initial_state, considered_models, profiles, parameters):
        num_model_options=0
        self.profiles=profiles
        self.parameters=parameters
        self.models=considered_models
        state_length=len(profiles)*dataSettings.nx+len(parameters)
        self.all_predictions=torch.zeros((len(self.models),state_length))
        for which_model,model in enumerate(self.models):
            self.all_predictions[which_model]=initial_state
    def warmup_step(self, input_tensor):
        for which_model,model in enumerate(self.models):
            self.all_predictions[which_model]=model(input_tensor[None,:])[0]
    def prediction_step(self, actuator_array):
        for which_model,model in enumerate(self.models):
            input_tensor=torch.cat((self.all_predictions[which_model],actuator_array))
            self.all_predictions[which_model]=model(input_tensor[None,:])[0]
    def get_denormed_predictions(self):
        normed_dic=dataSettings.state_to_dic(self.all_predictions, profiles=self.profiles, parameters=self.parameters)
        for sig in normed_dic:
            normed_dic[sig]=torch.stack(normed_dic[sig])
        return dataSettings.get_denormalized_dic(normed_dic)

def get_considered_models(config_filename, ensemble=True):
    config=configparser.ConfigParser()
    config.read(config_filename)
    output_filename_base=config['model']['output_filename_base']
    output_dir=config['model']['output_dir']
    model_type=config['model']['model_type']
    profiles=config['inputs']['profiles'].split()
    actuators=config['inputs']['actuators'].split()
    parameters=config['inputs']['parameters'].split()
    state_length=len(profiles)*dataSettings.nx+len(parameters)
    actuator_length=len(actuators)
    considered_models=[]
    if ensemble:
        all_model_files=glob.glob(os.path.join(output_dir, f'{output_filename_base}[0-9]*.tar'))
        # exclude models under the median loss
        losses = []
        for model_file in all_model_files:
            saved_state=torch.load(model_file, map_location=torch.device('cpu'))
            losses.append(np.min([saved_state['val_losses'][-i] for i in range(10)]))
        max_loss = np.median(losses)
        for model_file in all_model_files:
            saved_state=torch.load(model_file, map_location=torch.device('cpu'))
            if saved_state['val_losses'][-1]<max_loss:
                model=models[model_type](input_dim=state_length+2*actuator_length, output_dim=state_length,
                                         **saved_state['model_hyperparams'])
                model.load_state_dict(saved_state['model_state_dict'])
                considered_models.append(model)
        print(f'{len(considered_models)}/{len(all_model_files)} models used (i.e. only loss<{max_loss:0.2e})')
    else:
        model_file=os.path.join(output_dir, f'{output_filename_base}.tar')
        saved_state=torch.load(model_file, map_location=torch.device('cpu'))
        model=models[model_type](input_dim=state_length+2*actuator_length, output_dim=state_length,
                                 **saved_state['model_hyperparams'])
        model.load_state_dict(saved_state['model_state_dict'])
        considered_models=[model]
    return considered_models

def get_predictions(normalized_true_state, considered_models, profiles, parameters, nwarmup=0):
    predicted_means={}
    predicted_stds={}
    time_length=len(normalized_true_state)
    for profile in profiles:
        predicted_means[profile]=np.zeros((time_length,dataSettings.nx))
        predicted_stds[profile]=np.zeros((time_length,dataSettings.nx))
    for parameter in parameters:
        predicted_means[parameter]=np.zeros((time_length,1))
        predicted_stds[parameter]=np.zeros((time_length,1))
    all_predictions=get_predictions_per_model(normalized_true_state, considered_models, profiles, parameters, nwarmup=nwarmup)
    for sig in profiles+parameters:
        for step, denormed_prediction in enumerate(all_predictions[sig]):
            predicted_means[sig][step, :]=torch.mean(denormed_prediction, dim=0).detach().numpy()
            predicted_stds[sig][step, :]=torch.std(denormed_prediction, dim=0).detach().numpy()
    return predicted_means, predicted_stds

def get_predictions_per_model(normalized_true_state, considered_models, profiles, parameters, nwarmup=0):
    time_length=len(normalized_true_state)
    state_length=len(profiles)*dataSettings.nx+len(parameters)
    predicted_values = {}
    for profile in profiles:
        predicted_values[profile]=torch.zeros((time_length,len(considered_models),dataSettings.nx))
    for parameter in parameters:
        predicted_values[parameter]=torch.zeros((time_length,len(considered_models),1))
    for step in range(time_length):
        step_tensor=normalized_true_state[step]
        step_state=step_tensor[:state_length]
        step_actuators=step_tensor[state_length:]
        if step==0:
            modelstepper=ModelStepper(step_state, considered_models, profiles, parameters)
        if step<nwarmup:
            modelstepper.warmup_step(step_tensor)
        else:
            modelstepper.prediction_step(step_actuators)
        denormed_predictions=modelstepper.get_denormed_predictions()
        for sig in profiles+parameters:
            predicted_values[sig][step, :, :]=denormed_predictions[sig]
    return predicted_values

def get_fake_actuator_state(normalized_true_state, profiles, parameters, actuators):
    fake_actuator_state=normalized_true_state.clone()
    fake_actuator_dic=dataSettings.state_to_dic(fake_actuator_state, profiles=profiles, parameters=parameters, actuators=actuators)
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
    fake_actuator_state = dataSettings.dic_to_state(fake_actuator_dic, profiles, parameters, actuators=actuators)
    return fake_actuator_state
