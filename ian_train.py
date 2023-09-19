import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from customDatasetMakers import preprocess_data, ian_dataset
from customModels import IanRNN, IanMLP

from dataSettings import nx, train_shots, val_shots, test_shots, val_indices

import configparser
import os
import sys
import time

models={'IanRNN': IanRNN, 'IanMLP': IanMLP}

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename='configs/default.cfg'

config=configparser.ConfigParser()
config.read(config_filename)
preprocessed_data_filenamebase=config['preprocess']['preprocessed_data_filenamebase']
model_type=config['model']['model_type']
bucket_size=int(config['optimization']['bucket_size'])
n_epochs=int(config['optimization']['n_epochs'])
lr=float(config['optimization']['lr'])
lr_gamma=float(config['optimization']['lr_gamma'])
early_stopping=bool(config['optimization']['early_stopping'])
energyWeight=float(config['optimization']['energyWeight'])
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs']['parameters'].split()

model_hyperparams={key: int(val) for key,val in dict(config[model_type]).items()}

# dump to same location as the config filename, with .tar instead of .cfg
output_filename=os.path.join(config['model']['output_dir'],model_type+".tar")

print('Organizing train data from preprocessed_data')
start_time=time.time()
x_train, y_train, shots, times = ian_dataset(preprocessed_data_filenamebase+'train.pkl',
                                             profiles, actuators, parameters,
                                             sort_by_size=True)
print(f'...took {(time.time()-start_time):0.2f}s')
print('Organizing validation data from preprocessed_data')
start_time=time.time()
x_val, y_val, shots, times = ian_dataset(preprocessed_data_filenamebase+'val.pkl',
                                         profiles, actuators, parameters,
                                         sort_by_size=True)
print(f'...took {(time.time()-start_time):0.2f}s')

state_length=len(profiles)*33+len(parameters)
actuator_length=len(actuators)
model=models[model_type](input_dim=state_length+2*actuator_length, output_dim=state_length,
                         **model_hyperparams)

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

train_losses=[]
val_losses=[]
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50,70], gamma=lr_gamma, verbose=True)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma)

print('Training...')
if torch.cuda.is_available():
    device='cuda'
    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPU(s)")
else:
    device = 'cpu'
    print("Using CPU")
model.to(device)
start_time=time.time()
prev_time=start_time

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
    return buckets

train_x_buckets = make_bucket(x_train, bucket_size)
train_y_buckets = make_bucket(y_train, bucket_size)
train_length_buckets = [[len(arr) for arr in bucket] for bucket in train_x_buckets]

val_x_buckets = make_bucket(x_val, bucket_size)
val_y_buckets = make_bucket(y_val, bucket_size)
val_length_buckets = [[len(arr) for arr in bucket] for bucket in val_x_buckets]

avg_train_losses=[]
avg_val_losses=[]
for epoch in range(n_epochs):
    model.train()
    train_losses=[]
    for which_bucket in range(len(train_x_buckets)):
        x_bucket=train_x_buckets[which_bucket]
        y_bucket=train_y_buckets[which_bucket]
        length_bucket=train_length_buckets[which_bucket]
        # randomize within bucket for training
        # indices = torch.randperm(len(x_bucket))
        # x_bucket = [x_bucket[i] for i in indices]
        # y_bucket = [y_bucket[i] for i in indices]
        # length_bucket = [length_bucket[i] for i in indices]

        padded_x=pad_sequence(x_bucket, batch_first=True)
        padded_y=pad_sequence(y_bucket, batch_first=True)
        padded_x=padded_x.to(device)
        padded_y=padded_y.to(device)

        optimizer.zero_grad()
        model_output=model(padded_x) #, length_bucket)
        train_loss=masked_loss(loss_fn,
                               model_output, padded_y,
                               length_bucket)
        # Backpropagation
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())
    scheduler.step()
    avg_train_losses.append(sum(train_losses)/len(train_losses)) # now divide by total number of samples to get mean over steps/batches
    model.eval()
    val_losses=[]
    with torch.no_grad():
        for which_bucket in range(len(val_x_buckets)):
            x_bucket=val_x_buckets[which_bucket]
            y_bucket=val_y_buckets[which_bucket]
            length_bucket=val_length_buckets[which_bucket]
            padded_x=pad_sequence(x_bucket, batch_first=True)
            padded_y=pad_sequence(y_bucket, batch_first=True)
            padded_x=padded_x.to(device)
            padded_y=padded_y.to(device)
            model_output = model(padded_x) #, length_bucket)
            val_loss = masked_loss(loss_fn,
                                   model_output, padded_y,
                                   length_bucket)
            val_losses.append(val_loss.item())
        avg_val_losses.append(sum(val_losses)/len(val_losses))
    print(f'{epoch+1:4d}/{n_epochs}({(time.time()-prev_time):0.2f}s)... train: {avg_train_losses[-1]:0.2e}, val: {avg_val_losses[-1]:0.2e};')
    if early_stopping and avg_val_losses[-1]==min(avg_val_losses):
        print(f"Checkpoint")
        torch.save({
            'epoch': epoch,
            'val_indices': val_indices,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': avg_train_losses,
            'val_losses': avg_val_losses,
            'profiles': profiles,
            'actuators': actuators,
            'parameters': parameters,
            #'space_inds': space_inds,
            'exclude_ech': True
        }, output_filename)
    prev_time=time.time()

print(f'...took {(time.time()-start_time)/60:0.2f}min')
