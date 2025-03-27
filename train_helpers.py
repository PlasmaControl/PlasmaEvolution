import torch
import numpy as np
import dataSettings
from customDatasetMakers import get_state_indices_dic

# 2D mask (to matrix-transform the state)
def get_state_mask(profiles, parameters,
                   masked_outputs=[], rho_bdry_index=None,
                   nx=dataSettings.nx):
    indices_dic=get_state_indices_dic(profiles,parameters,calculations=[],nx=nx)
    state_length=0
    for sig in profiles+parameters:
        state_length+=len(np.atleast_1d(indices_dic[sig]))
    mask=torch.ones(state_length)
    masked_state_indices=[]
    for sig in masked_outputs:
        mask[indices_dic[sig]]=0
    if rho_bdry_index is not None:
        for sig in profiles:
            mask[indices_dic[sig][rho_bdry_index:]]=0
    return mask

# projects a state mask across samples and times
def get_sample_time_state_mask(state_mask, dimensions, lengths, nwarmup=0):
    # dimensions should be like (nsamples, ntimes, nstates)
    full_mask=torch.zeros(dimensions)
    for sample_index,length in enumerate(lengths):
        full_mask[sample_index,nwarmup:length,:]=state_mask[...]
    return full_mask

# loss function should sum over all values, we normalize ourselves here
# e.g. torch.nn.MSELoss(reduction='sum')
def masked_loss(loss_fn,
                output, target,
                mask):
    #mask=get_mask(output.size(), lengths, nwarmup, masked_state_indices)
    output=output*mask
    target=target*mask
    # normalize by dividing out number of included points
    return loss_fn(output, target) / (torch.count_nonzero(mask))

# make buckets of near-even size from a sorted array of arrays
def make_bucket(arrays, bucket_size):
    buckets=[]
    current_bucket=[]
    current_len=0
    for arr in arrays:
        arr_len=len(arr)
        current_bucket.append(arr)
        current_len+=arr_len
        if current_len > bucket_size:
            buckets.append(current_bucket)
            current_bucket=[]
            current_len=0
    if len(current_bucket)>0:
        buckets.append(current_bucket)
    return buckets

# calculate controllability of LRAN model for training loss
def get_controllability(model_type, model, eps=1e-6):
    """
    Compute a differentiable penalty that is large when the system
    is poorly controllable (small sigma_min of the controllability matrix).
    
    model: your HiroLRAN / HiroLRANDiag or similar, assumed to have:
           model.A.weight (d x d) and model.B.weight (d x m).
    alpha: scalar weight from config (controllability_weight).
    eps:   small offset to avoid division by zero.
    """
    # Extract A and B from the model. Make sure they require grad if we want them to be learned.
    if model_type == 'HiroLRAN_nondiag':
        A = model.A.weight     # shape (d, d)
        B = model.B.weight     # shape (d, m)
    elif model_type == 'HiroLRAN' or model_type == 'HiroLRANInverse':
        A = torch.diag(model.A.diagonal)
        B = model.B.weight
    d = A.shape[0]

    # Build controllability matrix: [B, A B, A^2 B, ..., A^(d-1) B]
    # We'll do an up-to-(d-1) expansion for discrete-time controllability of dimension d.
    blocks = []
    A_power = torch.eye(d, device=A.device, dtype=A.dtype)
    for _ in range(d):
        blocks.append(A_power @ B)
        A_power = A @ A_power
    C = torch.cat(blocks, dim=1)  # shape (d, d*m)

    # smallest singular value of C
    # NOTE: for small d, full SVD is fine. For bigger d, consider alternatives (e.g. truncated SVD).
    # Also note torch.svd is deprecated in newer PyTorch in favor of torch.linalg.svd
    U, S, V = torch.linalg.svd(C, full_matrices=False)
    sigma_min = S[-1]  # S is sorted in descending order

    # We want to maximize sigma_min => minimize 1 / sigma_min.
    controllability = sigma_min + eps
    return controllability

def get_controllability_A_B(A, B, eps=1e-6):
    """
    Compute a differentiable penalty that is large when the system is poorly
    controllable (i.e. when the smallest singular value of the controllability 
    matrix is small).

    Parameters:
      A (torch.Tensor): System dynamics matrix of shape (d, d).
      B (torch.Tensor): Input matrix of shape (d, m).
      eps (float): Small offset to avoid division by zero.

    Returns:
      torch.Tensor: sigma_min + eps, where sigma_min is the smallest singular value
                    of the controllability matrix [B, A@B, A^2@B, ..., A^(d-1)@B].
    """
    d = A.shape[0]

    # Build the controllability matrix: [B, A@B, A^2@B, ..., A^(d-1)@B]
    blocks = []
    A_power = torch.eye(d, device=A.device, dtype=A.dtype)
    for _ in range(d):
        blocks.append(A_power @ B)
        A_power = A @ A_power
    C = torch.cat(blocks, dim=1)  # Shape: (d, d*m)

    # Compute singular values and extract the smallest one.
    U, S, V = torch.linalg.svd(C, full_matrices=False)
    sigma_min = S[-1]

    # Return the penalty (larger when the system is poorly controllable).
    return sigma_min + eps