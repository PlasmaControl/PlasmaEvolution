import torch

def masked_loss(loss_fn,
                output, target,
                lengths,
                nwarmup=0,
                masked_indices=[]):
    loss_fn=torch.nn.MSELoss(reduction='sum')
    mask = torch.zeros_like(output) # nsamples, ntimes, nstates
    for i, length in enumerate(lengths):
        mask[i, nwarmup:length]=1
    if len(masked_indices)>0:
        mask[:,:,masked_indices]=0
    mask=mask.to(output.device)
    output=output*mask
    target=target*mask
    # normalize by dividing out true number of time samples in all batches
    # times the state size
    return loss_fn(output, target) / (sum(lengths)*output.size(-1))

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
