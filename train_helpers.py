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
