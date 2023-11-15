import torch

def get_mask(mask_size, lengths, nwarmup, masked_indices):
    mask=torch.zeros(mask_size) # nsamples, ntimes, nstates
    for i, length in enumerate(lengths):
        mask[i, nwarmup:length]=1
    if len(masked_indices)>0:
        mask[:,:,masked_indices]=0
    return mask

# loss function should sum over all values, we normalize ourselves here
# e.g. torch.nn.MSELoss(reduction='sum')
def masked_loss(loss_fn,
                output, target,
                lengths,
                nwarmup=0,
                masked_indices=[]):
    mask=get_mask(output.size(), lengths, nwarmup, masked_indices)
    mask=mask.to(output.device)
    output=output*mask
    target=target*mask
    # normalize by dividing out number of included points
    return loss_fn(output, target) / (torch.sum(mask))

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
