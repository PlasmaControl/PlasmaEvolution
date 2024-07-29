
import torch
import configparser
import control
from dataSettings import get_denormalized_dic, get_normalized_dic
from customModels import IanRNN, HiroLRAN
from train_helpers import get_state_mask, get_sample_time_state_mask, masked_loss
import numpy as np
from scipy.sparse import csr_matrix
import customDatasetMakers
import matplotlib.pyplot as plt
import prediction_helpers
import osqp
import scipy as sp
from scipy import sparse


lstm_model_name = 'HiroLRAN_betan6'
#lstm_model_name = 'HiroLRAN_alldiiid'
linear_model_name = 'HiroLRAN_betan6'

config_filename = f'/projects/EKOLEMEN/profile_predictor/joe_hiro_models/{lstm_model_name}config'
config=configparser.ConfigParser()
config.read(config_filename)
output_filename_base=config['model']['output_filename_base']
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs'].get('parameters','').split()
calculations=config['inputs'].get('calculations','').split()
linear_config_filename = f'/projects/EKOLEMEN/profile_predictor/joe_hiro_models/{linear_model_name}config'
config_linear=configparser.ConfigParser()
config_linear.read(linear_config_filename)
controller_actuators = config_linear['inputs']['actuators'].split()
controller_profiles = config_linear['inputs']['profiles'].split()
controller_parameters = config_linear['inputs']['parameters'].split()
latent_dim = int(config_linear['HiroLRAN']['latent_dim'])
data_filename = config['preprocess']['preprocessed_data_filenamebase'] + 'val.pkl'

lstm_model = prediction_helpers.get_considered_models(config_filename, ensemble=False)[0]

linear_model = prediction_helpers.get_considered_models(linear_config_filename, ensemble=False)[0]

x_test, y_test, shots, times =customDatasetMakers.ian_dataset(data_filename,profiles,parameters,calculations,actuators,sort_by_size=True)

shot_index = 10
wanted_sample = x_test[shot_index]
nwarmup = 3
starting_index = 0
end_index = len(wanted_sample) - nwarmup

N = 10 # prediction horizon
nsim = 250 # number of simulated timesteps

# get the simulator indices of the states and actuators that I want to control
future_controller_indices = [len(profiles)*33 + len(parameters) + len(actuators) + actuators.index(controller_actuators[i]) for i in range(len(controller_actuators))]
current_controller_indices = [len(profiles)*33 + len(parameters) + actuators.index(controller_actuators[i]) for i in range(len(controller_actuators))]
initial_state_index_list = [profiles.index(profiles[i]) for i in range(len(controller_profiles))]
state_indices = [num * 33 + i for num in initial_state_index_list for i in range(33)]
parameter_indices = [len(profiles)*33 + parameters.index(controller_parameters[i]) for i in range(len(controller_parameters))]
state_indices = state_indices + parameter_indices

# the state that I alter throughout the simulation
simulated_state = wanted_sample[starting_index:end_index, :].clone()
simulated_state = torch.unsqueeze(simulated_state, 0).float()
simulated_z_t = linear_model.encoder(simulated_state[:,nwarmup,state_indices].clone()).detach().numpy()

target_params = wanted_sample[starting_index:end_index,state_indices].clone()
target_params = torch.unsqueeze(target_params, 0).float()
target_params = target_params * 0 + 3
target_z_t = linear_model.encoder(target_params).detach().numpy()

Q_weights = np.ones(latent_dim)
Q = np.eye(latent_dim)
for i in range(latent_dim):
    Q[i,i] = Q_weights[i]
Q = csr_matrix(Q)

QN = Q
R_array = np.eye(len(controller_actuators))*0.1
#R_array[controller_actuators.index('D_tot'), controller_actuators.index('D_tot')] = 0.01
R = csr_matrix(R_array)

# - linear constraints
umin = np.zeros(len(controller_actuators))
umax = np.ones(len(controller_actuators))*7
xmin = np.array([-np.inf]*latent_dim)
xmax = np.array([np.inf]*latent_dim)

# set actuators that I don't want to control
#blocked_actuators = ['tinj', 'ip', 'bt', 'D_tot', 'ech_pwr_total', 'tribot_EFIT01', 'tritop_EFIT01', 'kappa_EFIT01', 'aminor_EFIT01', 'volume_EFIT01', 'rmaxis_EFIT01']
blocked_actuators = []
for actuator in blocked_actuators:
    true_value = wanted_sample[starting_index, future_controller_indices[controller_actuators.index(actuator)]].clone().item()
    umin[controller_actuators.index(actuator)] = true_value
    umax[controller_actuators.index(actuator)] = true_value

Ad = sparse.csc_matrix(linear_model.A.weight.data)
Bd = sparse.csc_matrix(linear_model.B.weight.data)
[nx, nu] = Bd.shape

q = np.hstack([np.kron(np.ones(N), -Q@(target_z_t[0,nwarmup,:])), -QN@(target_z_t[0,nwarmup,:]), np.zeros(N*nu)])
P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])
leq = np.hstack([-simulated_z_t[0], np.zeros(N*nx)])
ueq = leq
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
A = sparse.vstack([Aeq, Aineq], format='csc')
l = np.hstack([leq, lineq])
u = np.hstack([ueq, uineq])

prob = osqp.OSQP()
prob.setup(P, q, A, l, u, warm_start=True, max_iter=10000)
controlled_profiles = []
actuator_trajectory = []
latent_trajectory = []

for i in range(nsim):
    # Solve
    res = prob.solve()

    # Check solver status
    if res.info.status != 'solved':
        raise ValueError('OSQP did not solve the problem!')
    
    ctrl = torch.tensor(res.x[-N*nu:-(N-1)*nu]).float()

    #x=res.x
    #obj_val = 0.5 * np.dot(x, P.dot(x)) + np.dot(q, x)
    #print(f'obj')
    #print(obj_val)
    #print(res.x)

    # change future actuator, and the current actuator of future timestep
    simulated_state[0,nwarmup+i,future_controller_indices] = ctrl
    simulated_state[0, nwarmup+i+1, current_controller_indices] = ctrl

    predicted_state = prediction_helpers.get_fast_profile_prediction(simulated_state[:,:nwarmup + i + 1, :], lstm_model)
    if nwarmup + i + 1 < len(wanted_sample):
        simulated_state[:, nwarmup + i + 1, state_indices] = predicted_state
    
    controlled_profiles.append(predicted_state.detach().numpy())
    actuator_trajectory.append(ctrl.detach().numpy())

    predicted_z_t = linear_model.encoder(predicted_state).detach().numpy()
    latent_trajectory.append(predicted_z_t)
    # Update limits
    for actuator in blocked_actuators:
        true_value = wanted_sample[starting_index+i, future_controller_indices[controller_actuators.index(actuator)]].clone().item()
        umin[controller_actuators.index(actuator)] = true_value
        umax[controller_actuators.index(actuator)] = true_value

    lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
    uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
    l = np.hstack([leq, lineq])
    u = np.hstack([ueq, uineq])

    l[:nx] = -predicted_z_t
    u[:nx] = -predicted_z_t

    # Update targets
    q = np.hstack([np.kron(np.ones(N), -Q@(target_z_t[0,nwarmup + i,:])), -QN@(target_z_t[0,nwarmup + i,:]), np.zeros(N*nu)])
    
    prob.update(l=l, u=u, q=q)
    print(f'timestep {i} done')

output_dict = {}
real_state = wanted_sample[starting_index:end_index,:].clone()
real_dict = customDatasetMakers.state_to_dic(real_state, profiles, parameters, actuators=actuators)
real_denormed_dict = get_denormalized_dic(real_dict)

controlled_state = simulated_state
controlled_dict = customDatasetMakers.state_to_dic(controlled_state, controller_profiles, controller_parameters, actuators=controller_actuators)
controlled_denormed_dict = get_denormalized_dic(controlled_dict)

output_dict['real'] = real_denormed_dict
output_dict['controlled'] = controlled_denormed_dict
output_dict['latent'] = np.array(latent_trajectory)

import pickle
with open(f'control_pickles/{lstm_model_name}{linear_model_name}{shot_index}.pkl', 'wb') as file:
    # Pickle the array and write it to the file
    pickle.dump(output_dict, file)
