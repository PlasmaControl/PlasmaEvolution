import torch
import customDatasetMakers
import customModels
from torch.utils.data import DataLoader
from dataSettings import nx, normalizations

import matplotlib.pyplot as plt
from torchviz import make_dot

input_filename='PlasmaConv2D.tar'
data_filename='test.h5'

saved_state=torch.load(input_filename)
model=customModels.PlasmaConv2D(saved_state['profiles'], saved_state['actuators'], saved_state['parameters'])
model.load_state_dict(saved_state['model_state_dict'])

dataset=customDatasetMakers.standard_dataset(data_filename,saved_state['profiles'],saved_state['actuators'],saved_state['parameters'],
                                             saved_state['lookahead'],saved_state['lookback'])
data_loader=DataLoader(dataset, batch_size=20)
output_profiles, input_profiles, input_actuators, input_parameters = next(iter(data_loader))
yhat=model(input_profiles, input_actuators, input_parameters)
make_dot(yhat).render("model",format="png")

if False:
    output_profiles_hat_list=[]
    output_profiles_list=[]
    model.eval()
    with torch.no_grad():
        for output_profiles, input_profiles, input_actuators, input_parameters in data_loader:
            output_profiles_list.append(output_profiles)
            output_profiles_hat_list.append(model(input_profiles, input_actuators, input_parameters))
