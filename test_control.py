import osqp
import pickle
import torch
import configparser
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import csr_matrix

import control
import matplotlib.pyplot as plt

# Local imports
import customDatasetMakers
import prediction_helpers
from dataSettings import get_denormalized_dic, get_normalized_dic
from customModels import IanRNN, HiroLRAN, HiroLRANInverse, HiroLRANDiag, HiroLRAN_nondiag
from train_helpers import get_state_mask, get_sample_time_state_mask, masked_loss
from prediction_helpers import (
    get_ml_truth, get_ml_profile_warmup, get_ml_actuator_trajectory,
    get_ml_predictions, get_considered_models, get_fast_profile_prediction,
    get_control_targets_from_profiles
)

# ============================================================================
# 1) Pick the Model and Load the Data
# ============================================================================

# load the models
true_model_name = 'IanRNN_v12'
control_model = 'HiroLRAN_v'
model_type = ''
if 'HiroLRAN' in true_model_name:
    model_type = 'HiroLRAN'
elif 'IanRNN' in true_model_name:
    model_type = 'IanRNN'

config_file = f'/projects/EKOLEMEN/profile_predictor/joe_hiro_models/{true_model}config'
config = configparser.ConfigParser()
config.read(config_file)

profiles    = config['inputs']['profiles'].split()
actuators   = config['inputs']['actuators'].split()
parameters  = config['inputs'].get('parameters','').split()
calculations= config['inputs'].get('calculations','').split()

linear_config_file = f'/projects/EKOLEMEN/profile_predictor/joe_hiro_models/{control_model}config'
config_linear = configparser.ConfigParser()
config_linear.read(linear_config_file)

controller_actuators  = config_linear['inputs']['actuators'].split()
controller_profiles   = config_linear['inputs']['profiles'].split()
controller_parameters = config_linear['inputs']['parameters'].split()

data_file = config['preprocess']['preprocessed_data_filenamebase'] + 'train.pkl'

true_model   = get_considered_models(config_file, ensemble=False)[0]
linear_model = get_considered_models(linear_config_file, ensemble=False)[0]

x_test, y_test, shots, times = customDatasetMakers.ian_dataset(
    data_file, profiles, parameters, calculations, actuators, sort_by_size=True
)



# ============================================================================
# 2) Set Simulation Parameters
# ============================================================================

shot_index     = 500
version        = 4
wanted_sample  = x_test[shot_index]
nwarmup        = 3
start_idx      = 100
end_idx        = len(wanted_sample) - nwarmup

N    = 10    # prediction horizon
nsim = 100   # simulation steps

if nsim > end_idx - start_idx - nwarmup: # can't simulate beyond the end of the data
    nsim = end_idx - start_idx - nwarmup

A = linear_model.A.weight.data.detach().numpy()
B = linear_model.B.weight.data.detach().numpy()

nz = A.shape[0]   # state dimension
nu = B.shape[1]   # control dimension

future_ctrl_idxs = [
    n_profiles*profile_size + n_parameters + n_acts
    + actuators.index(act) for act in controller_actuators
]
current_ctrl_idxs = [
    n_profiles*profile_size + n_parameters
    + actuators.index(act) for act in controller_actuators
]

state_idx_list = [profiles.index(p) for p in controller_profiles]
state_indices  = [p_idx*profile_size + i for p_idx in state_idx_list for i in range(profile_size)]
param_indices  = [n_profiles*profile_size + parameters.index(p)
                  for p in controller_parameters]
state_indices += param_indices

# define simulation state and initial state
sim_state = wanted_sample[start_idx:end_idx].clone().unsqueeze(0).float()

init_z     = linear_model.encoder(sim_state[:, nwarmup, state_indices]).detach().numpy()

manual_targets = True

if manual_targets:
    target_profiles = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0],
    ])
    profile_switch_times = np.array([0, 50])
    target_trajectory = get_control_targets_from_profiles(target_profiles, profile_switch_times, nsim)
    target_z = linear_model.encoder(torch.tensor(target_trajectory, dtype=torch.float32)).detach().numpy().flatten()
else:
    # use the real data as the targets
    target_z = linear_model.encoder(torch.tensor(wanted_sample[start_idx:start_idx+nsim, state_indices], dtype=torch.float32)).detach().numpy().flatten()

nx_orig = nz  # original state dimension

# ============================================================================
# 2) Build the Augmented System (with Integral Action)
# ============================================================================

# Augment the state: x = [ z ; e ] where e is the integral (tracking error).
A_aug = np.block([
    [A,                np.zeros((nx_orig, nx_orig))],
    [np.eye(nx_orig),  np.eye(nx_orig)]
])
B_aug = np.block([
    [B],
    [np.zeros((nx_orig, nu))]
])
nx_aug = 2 * nx_orig  # augmented state dimension

# ============================================================================
# 3) Define the Costs and Constraints
# ============================================================================

# Cost weights:
Q   = 1 * np.eye(nz)       # cost on the state error (z)
Q_e = 0.1 * np.eye(nz)       # cost on the integral error (e)
R   = 0.001 * np.eye(nu)             # cost on the control input

# Build the augmented cost matrix:
Q_aug = np.block([
    [Q,              np.zeros((nx_orig, nx_orig))],
    [np.zeros((nx_orig, nx_orig)), Q_e]
])
Q_aug_csr = csr_matrix(Q_aug)
R_csr     = csr_matrix(R)

# Normalized control constraints [ip, pinj, tinj, echp, PCBCOIL, gasA]: 
umin = np.array([
    (0 - 989467)/389572,
    (3 - 4.072876)/3.145593,
    (0 - 3.38)/2.70,
    0e6/1e6,
    1/58800,
    (0 - 0.2318070580561956)/1.6204600868125758
])
umax = np.array([
    (1e7 - 989467)/389572,
    (15 - 4.072876)/3.145593,
    (12 - 3.38)/2.70,
    3e6/1e6,
    3/58800,
    (10 - 0.2318070580561956)/1.6204600868125758
])

# Force 'blocked_actuators' to remain uncontrolled:
blocked_actuators = ['ip', 'PCBCOIL']
for ba in blocked_actuators:
    val = wanted_sample[start_idx+nwarmup, current_ctrl_idxs[controller_actuators.index(ba)]].item()
    idx = controller_actuators.index(ba)
    umin[idx] = val
    umax[idx] = val

# State constraints are unconstrained:
lineq_x = np.hstack([np.full(nx_orig, -np.inf),
                     np.full(nx_orig, -np.inf)])
uineq_x = np.hstack([np.full(nx_orig,  np.inf),
                     np.full(nx_orig,  np.inf)])

# Build the block-diagonal cost for the QP:
# There are N+1 state blocks and N control blocks.
P = sparse.block_diag([
    sparse.kron(sparse.eye(N+1), Q_aug_csr),
    sparse.kron(sparse.eye(N), R_csr)
], format='csc')

# ============================================================================
# 4) Build Dynamics and Constraint Matrices for OSQP
# ============================================================================

# Dynamics equality constraints:
#   x_{k+1} = A_aug * x_k + B_aug * u_k
# can be written as: -x_{k+1} + A_aug * x_k + B_aug * u_k = 0.
Ax = (sparse.kron(sparse.eye(N+1), -sparse.eye(nx_aug)) +
      sparse.kron(sparse.eye(N+1, k=-1), csr_matrix(A_aug)))
Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]),
                 csr_matrix(B_aug))
Aeq = sparse.hstack([Ax, Bu], format='csc')

# Build the equality constraint vector (will be updated in the loop):
leq = np.zeros(( (N+1)*nx_aug, ))
ueq = np.zeros(( (N+1)*nx_aug, ))

# Initial condition: x0_aug = [init_z; 0]
x0_aug = np.concatenate([init_z, np.zeros(nx_orig)])
leq[:nx_aug] = -x0_aug
ueq[:nx_aug] = -x0_aug

# For each future step, add an offset to enforce tracking:
# The offset is [0; -target_z] for the integral part.
for k in range(N):
    offset = np.zeros(nx_aug)
    offset[nx_orig:] = -target_z[0]  # initially use the first target value
    idx_start = (k+1)*nx_aug
    idx_end   = (k+2)*nx_aug
    leq[idx_start:idx_end] = offset
    ueq[idx_start:idx_end] = offset

# Inequality constraints:
Aineq = sparse.eye((N+1)*nx_aug + N*nu)
lineq = np.hstack([np.kron(np.ones(N+1), lineq_x),
                   np.kron(np.ones(N), umin)])
uineq = np.hstack([np.kron(np.ones(N+1), uineq_x),
                   np.kron(np.ones(N), umax)])

# Stack equality and inequality constraints:
A_ = sparse.vstack([Aeq, Aineq], format='csc')
l_ = np.hstack([leq, lineq])
u_ = np.hstack([ueq, uineq])

# Build the linear term of the cost function:
xref_tmp = np.concatenate([target_z[0], np.zeros(nx_orig)])
q_step = -Q_aug @ xref_tmp
q_vec_list = []
for k in range(N):
    q_vec_list.extend(q_step)
q_vec_list.extend(q_step)  # final state block
q_vec_list.extend(np.zeros(N*nu))  # no cost on inputs
q_vec = np.array(q_vec_list)

# ============================================================================
# 5) Setup OSQP and Run the MPC Simulation Loop
# ============================================================================

prob = osqp.OSQP()
prob.setup(P, q_vec, A_, l_, u_, warm_start=True, max_iter=10000)

# Storage for trajectories:
x_traj = [x0_aug]  # list of augmented states
#true_traj = []     # decoded states history
true_latent_traj = []  # encode decode state history, so the 'measured' latent value
u_traj = []        # control inputs history
objective_vals = []

for i in range(nsim):
    # Solve the QP
    res = prob.solve()
    if res.info.status != 'solved':
        raise ValueError("OSQP did not solve the problem!")
    objective_vals.append(res.info.obj_val)
    
    # Decision variables: first (N+1)*nx_aug for states, next N*nu for controls.
    n_state_vars = (N+1)*nx_aug
    u_sol = res.x[n_state_vars:]
    ctrl0 = u_sol[:nu]  # extract the first control input
    u_traj.append(ctrl0)
    
    # update simulated state with new actuation
    sim_state[0, nwarmup + i - 1, future_ctrl_idxs] = ctrl0
    sim_state[0, nwarmup + i, current_ctrl_idxs]    = ctrl0

    # Simulate one step:
    if model_type == 'HiroLRAN':
        x_next = A_aug @ x_traj[-1] + B_aug @ ctrl0
    elif model_type == 'IanRNN':
        pred_state = get_fast_profile_prediction(sim_state[:, :nwarmup + i, :], lstm_model)
        x_next = linear_model.encoder(torch.tensor(pred_state, dtype=torch.float32)).detach().numpy().flatten()
        if (nwarmup + i) < len(wanted_sample):
            sim_state[:, nwarmup + i, :n_profiles*profile_size + n_parameters] = pred_state

    x_traj.append(x_next)
    z_next = x_next[:nz]

    true_next = linear_model.decoder(torch.tensor(z_next.reshape(1,1,len(z_next)), dtype=torch.float32))
    true_latent = linear_model.encoder(true_next).detach().numpy().flatten()

    #true_traj.append(true_next.detach().numpy().flatten())
    true_latent_traj.append(true_latent)

    # expand true_latent
    true_latent = np.concatenate([true_latent, x_next[nz:]])
    #true_latent = x_next #REMOVE THIS
    # the proper next initial condition is encode(decode(x_next))
    # Update the initial condition in the equality constraints:
    l_[:nx_aug] = -true_latent
    u_[:nx_aug] = -true_latent
    
    # Update the offsets for future steps using the target trajectory.
    for k in range(N):
        offset = np.zeros(nx_aug)
        t_idx = min(i + k + 1, target_z.shape[0]-1)
        offset[nx_orig:] = -target_z[t_idx]
        idx_start = (k+1)*nx_aug
        idx_end   = (k+2)*nx_aug
        l_[idx_start:idx_end] = offset
        u_[idx_start:idx_end] = offset
    
    # Update the linear cost vector q_vec for the updated horizon.
    q_vec_list = []
    for k in range(N):
        t_idx = min(i + k + 1, target_z.shape[0]-1)
        xref_tmp = np.concatenate([target_z[t_idx], np.zeros(nx_orig)])
        step_q = -Q_aug @ xref_tmp
        q_vec_list.extend(step_q)
    t_idx = min(i + N + 1, target_z.shape[0]-1)
    xref_tmp = np.concatenate([target_z[t_idx], np.zeros(nx_orig)])
    final_q = -Q_aug @ xref_tmp
    q_vec_list.extend(final_q)
    q_vec_list.extend(np.zeros(N*nu))
    q_vec = np.array(q_vec_list)
    
    prob.update(l=l_, u=u_, q=q_vec)

# ============================================================================
# 6) Save the Results
# ============================================================================

# Extract the latent state (first nz entries of each augmented state)
controlled_latent_traj = np.array([x[:nz] for x in x_traj])
controlled_traj = linear_model.decoder(torch.tensor(controlled_latent_traj, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
# Slice the target trajectory to match the simulation length
target_latent_traj = target_z[:len(controlled_latent_traj)]
target_traj = linear_model.decoder(torch.tensor(target_latent_traj, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
actuator_values = np.array(u_traj)
true_traj = linear_model.decoder(torch.tensor(np.array(true_latent_traj), dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
true_latent_traj = np.array(true_latent_traj)

output_dict = {
    'controlled_latent_traj': controlled_latent_traj,  # controlled latent trajectory (what the MPC intended)
    'controlled_traj': controlled_traj,                # controlled trajectory
    'target_latent_traj': target_latent_traj,            # target latent trajectory
    'target_traj': target_traj,                # target trajectory
    'true_traj': true_traj,                    # true_trajectory (what the MPC actually did)
    'true_latent_traj': true_latent_traj,      # true latent trajectory
    'actuator_values': actuator_values,        # control inputs applied
    'objective_vals': objective_vals           # OSQP objective history
}

save_path = 'control_pickles/control_results.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(output_dict, f)

print("MPC simulation complete, results saved to", save_path)
