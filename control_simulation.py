
import torch
import configparser
import control
from dataSettings import get_denormalized_dic, get_normalized_dic
from customModels import IanRNN, HiroLinear, HiroLRAN
from train_helpers import get_state_mask, get_sample_time_state_mask, masked_loss
import numpy as np
from scipy.sparse import csr_matrix
import customDatasetMakers
import matplotlib.pyplot as plt
import prediction_helpers
import osqp
import scipy as sp
from scipy import sparse


lstm_model_name = 'alldiiid_ensemble'
#lstm_model_name = 'HiroLRAN_alldiiid'
linear_model_name = 'HiroLRAN_alldiiid'

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

index = 21
wanted_sample = x_test[index]
nwarmup = 3
starting_index = 0
end_index = len(wanted_sample) - nwarmup - 200
nsim = end_index - starting_index - nwarmup 

# get the simulator indices of the states and actuators that I want to control
controller_indices = [len(profiles)*33 + len(parameters) + len(actuators) + actuators.index(controller_actuators[i]) for i in range(len(controller_actuators))]
initial_state_index_list = [profiles.index(profiles[i]) for i in range(len(controller_profiles))]
state_indices = [num * 33 + i for num in initial_state_index_list for i in range(33)]
parameter_indices = [len(profiles)*33 + parameters.index(controller_parameters[i]) for i in range(len(controller_parameters))]
state_indices = state_indices + parameter_indices

# the state that I alter throughout the simulation
simulated_state = wanted_sample[starting_index:end_index, :].clone()
simulated_state = torch.unsqueeze(simulated_state, 0).float()

simulated_z_t = linear_model.encoder(simulated_state[:,nwarmup,state_indices].clone()).detach().numpy()
#wanted_sample[75:-1, list(range(33*2))] = wanted_sample[150, list(range(33*2))].clone() * 1.5

#blocked target parameters
target_params = wanted_sample[starting_index:end_index,state_indices].clone()
#target_params[75:-1,state_indices] = wanted_sample[150,state_indices].clone()

target_params = torch.unsqueeze(target_params, 0).float()

target_z_t = linear_model.encoder(target_params).detach().numpy()

Q_weights = [0.8100542971289384, 0.47733788133334454, 1.380290183825432, 0.1946600370096445, 0.8371611953412182, 0.2929893921637235, 1.1841541784604726, 0.5459019292017557, 0.29055378627390116, 0.4864184396052758, 1.0151097353560643, 0.7351499604232643, 1.511569434309833, 0.20119677018643076, 0.7748890588570014, 0.2875803567369798, 0.4371611764390748, 0.19867922470886415, 4116.9207652067225, 0.17226876053190635, 0.411901216473951, 0.19979134416485317, 0.31474484740221126, 0.1502031498977518, 0.3967957423894341, 0.694605883617399, 0.4296677937543288, 0.8542615989583408, 1.2841927907016975, 0.8524922361045352, 0.19405377925886638, 0.24436839833659757, 0.1593508742349224, 0.43689683648784566, 0.40003578506149173, 0.3561962660530412, 0.1533393231506439, 0.33160692548105414, 0.20306547578413953, 0.9006677750619647]
Q = np.eye(latent_dim)
for i in range(latent_dim):
    Q[i,i] = Q_weights[i]
Q = csr_matrix(Q)

QN = Q
R_array = np.eye(len(controller_actuators))*0.01
R_array[controller_actuators.index('D_tot'), controller_actuators.index('D_tot')] = 0.01
R = csr_matrix(R_array)
N = end_index - starting_index - nwarmup
N = 10
# - linear constraints
umin = np.zeros(len(controller_actuators))
umax = np.ones(len(controller_actuators))*7
umax[controller_actuators.index('D_tot')] = 1
# set actuators bound by true values
blocked_actuators = ['ip', 'bt','tribot_EFIT01', 'tritop_EFIT01', 'kappa_EFIT01', 'aminor_EFIT01', 'volume_EFIT01', 'rmaxis_EFIT01']
for actuator in blocked_actuators:
    true_value = wanted_sample[starting_index, controller_indices[controller_actuators.index(actuator)]].clone().item()
    umin[controller_actuators.index(actuator)] = true_value
    umax[controller_actuators.index(actuator)] = true_value

xmin = np.array([-np.inf]*latent_dim)
xmax = np.array([np.inf]*latent_dim)

Ad = sparse.csc_matrix(linear_model.A.weight.data)
Bd = sparse.csc_matrix(linear_model.B.weight.data)
[nx, nu] = Bd.shape
# - linear objective

# set the target
q = np.hstack([np.kron(np.ones(N), -Q@(target_z_t[0,nwarmup,:])), -QN@(target_z_t[0,nwarmup,:]), np.zeros(N*nu)])

P = sparse.block_diag([sparse.kron(sparse.eye(N), Q), QN,
                       sparse.kron(sparse.eye(N), R)], format='csc')
Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
Aeq = sparse.hstack([Ax, Bu])

leq = np.hstack([-simulated_z_t[0], np.zeros(N*nx)])
ueq = leq
# - input and state constraints
Aineq = sparse.eye((N+1)*nx + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
# - OSQP constraints
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
    # Apply first control input to the plant
    ctrl = torch.tensor(res.x[-N*nu:-(N-1)*nu]).float()

    simulated_state[0,nwarmup+i,controller_indices] = ctrl

    predicted_state = lstm_model(simulated_state[:,:nwarmup + i + 1, :], nwarmup=nwarmup)

    controlled_profiles.append(predicted_state[0, nwarmup + i, state_indices].detach().numpy())
    actuator_trajectory.append(ctrl.detach().numpy())

    predicted_z_t = linear_model.encoder(torch.unsqueeze(predicted_state[:,nwarmup+i,state_indices],0)).detach().numpy()
    latent_trajectory.append(predicted_z_t)
    # Update limits
    for actuator in blocked_actuators:
        true_value = wanted_sample[starting_index+i, controller_indices[controller_actuators.index(actuator)]].clone().item()
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

controlled_state = np.hstack((np.array(controlled_profiles), np.array(actuator_trajectory), np.array(actuator_trajectory)))
controlled_dict = customDatasetMakers.state_to_dic(controlled_state, controller_profiles, controller_parameters, actuators=controller_actuators)
controlled_denormed_dict = get_denormalized_dic(controlled_dict)

output_dict['real'] = real_denormed_dict
output_dict['controlled'] = controlled_denormed_dict
output_dict['latent'] = latent_trajectory
import pickle

with open(f'control_pickles/{lstm_model_name}{linear_model_name}{index}.pkl', 'wb') as file:
    # Pickle the array and write it to the file
    pickle.dump(output_dict, file)
