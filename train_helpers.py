import torch

def masked_loss(loss_fn,
                output, target,
                lengths):
    mask = torch.zeros(len(lengths), max(lengths))
    for i, length in enumerate(lengths):
        mask[i, :length]=1
    mask=mask.to(output.device)
    output=output*mask[..., None]
    target=target*mask[..., None]
    # normalize by dividing out true number of time samples in all batches
    # times the state size
    return loss_fn(output, target) / (sum(lengths)*output.size(-1))

# I divide out by myself since different sequences/batches have different sizes
loss_fn=torch.nn.MSELoss(reduction='sum')

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
