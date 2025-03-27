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
    get_ml_predictions, get_considered_models, get_fast_profile_prediction
)

# ============================================================================
# 1) Define the 3x3 System and Simulation Parameters
# ============================================================================

# load the model
model_name = 'HiroLRAN_v23'

config_file = f'/projects/EKOLEMEN/profile_predictor/joe_hiro_models/{model_name}config'
config = configparser.ConfigParser()
config.read(config_file)

data_file = config['preprocess']['preprocessed_data_filenamebase'] + 'train.pkl'

model   = get_considered_models(config_file, ensemble=False)[0]

# Highly controllable system:
A = np.diag([0.9, 0.8, 0.7])   # 3x3 diagonal dynamics matrix
B = np.eye(3)                  # 3x3 identity (full rank) control matrix

A = torch.diag(model.A.diagonal.data).detach().numpy()
#A = model.A.weight.data.detach().numpy()
B = model.B.weight.data.detach().numpy()

nz = A.shape[0]   # state dimension (3)
nu = B.shape[1]   # control dimension (3)

# Initial latent state (z) and target trajectory:
init_z = np.array([-0.6202, -0.2201, -0.8403, -0.0348, -0.1184])
#init_z = np.ones((nz,))  # initial latent state
T = 150  # total time steps for the target trajectory
# For example, ramp trajectories for each state dimension:
target_z = np.vstack([np.linspace(0, 1, T),
                      np.linspace(0, -1, T),
                      np.linspace(0, 0.5, T)]).T  # shape: (T, 3)

target_z = np.ones((T, nz))  # constant target trajectory
# target_z is just init_z copied T times
target_z = np.tile([[-0.5861, -0.4475, -0.7799, -0.2300, -0.0227]], (T, 1))
#target_z = np.zeros((T, nz))  # constant target trajectory

# second half of target_z is init_z copied T//2 times
target_z[T//2:] = np.tile(init_z, (T//2, 1))
# MPC parameters:
N    = 10    # prediction horizon
nsim = 100   # simulation steps (nsim <= T)

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
# 3) Define the Cost and Constraints
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

# Control constraints: u in [-10, 10] for each actuator.
umin = -10 * np.ones(nu)
umax =  10 * np.ones(nu)

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
    
    # Simulate one step:
    x_next = A_aug @ x_traj[-1] + B_aug @ ctrl0
    x_traj.append(x_next)
    z_next = x_next[:nz]

    true_next = model.decoder(torch.tensor(z_next.reshape(1,1,len(z_next)), dtype=torch.float32))
    true_latent = model.encoder(true_next).detach().numpy().flatten()

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
controlled_traj = model.decoder(torch.tensor(controlled_latent_traj, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
# Slice the target trajectory to match the simulation length
target_latent_traj = target_z[:len(controlled_latent_traj)]
target_traj = model.decoder(torch.tensor(target_latent_traj, dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
actuator_values = np.array(u_traj)
true_traj = model.decoder(torch.tensor(np.array(true_latent_traj), dtype=torch.float32).unsqueeze(0)).detach().numpy()[0]
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
