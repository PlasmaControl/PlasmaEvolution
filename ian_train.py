import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from customDatasetMakers import preprocess_data, ian_dataset
from customModels import IanRNN, IanMLP, HiroLinear
from train_helpers import make_bucket, masked_loss

from dataSettings import nx, train_shots, val_shots, test_shots, val_indices

import configparser
import os
import sys
import time

models={'IanRNN': IanRNN, 'IanMLP': IanMLP, 'HiroLinear': HiroLinear}

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename='configs/default.cfg'

config=configparser.ConfigParser()
config.read(config_filename)
preprocessed_data_filenamebase=config['preprocess']['preprocessed_data_filenamebase']
model_type=config['model']['model_type']
bucket_size=config['optimization'].getint('bucket_size')
nwarmup=config['optimization'].getint('nwarmup')
n_epochs=config['optimization'].getint('n_epochs')
lr=config['optimization'].getfloat('lr')
lr_gamma=config['optimization'].getfloat('lr_gamma')
lr_stop_epoch=config['optimization'].getint('lr_stop_epoch')
early_saving=config['optimization'].getboolean('early_saving')
l1_lambda=config['optimization'].getfloat('l1_lambda')
l2_lambda=config['optimization'].getfloat('l2_lambda')
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs']['parameters'].split()
autoregression_num_steps=config['optimization'].getfloat('autoregression_num_steps',1)
autoregression_start_epoch=config['optimization'].getint('autoregression_start_epoch',int(n_epochs/4))
autoregression_end_epoch=config['optimization'].getint('autoregression_end_epoch',int(3*n_epochs/4))
if autoregression_num_steps<1:
    autoregression_num_steps=1
if 'tuning' in config:
    tune_model=config['tuning'].getboolean('tune_model',False)
    if tune_model:
        if 'model_to_tune_filename_base' not in config['tuning']:
            raise Exception("config['tuning']['tune_model'] set to true but no starting file specified in config['tuning']['model_to_tune_filename_base']")
        model_to_tune_filename_base=config['tuning']['model_to_tune_filename_base']
    frozen_layers=config['tuning'].get('frozen_layers','').split()
    resume_training=config['tuning'].getboolean('resume_training',False)
else:
    tune_model=False
# epoch to start on, should be 0 generally but can increase w/ tune_model to restart a model that stopped halfway
# at the moment, by default tune_model will start the epochs where the previous left off
start_epoch=0

model_hyperparams={key: int(val) for key,val in dict(config[model_type]).items()}

state_length=len(profiles)*33+len(parameters)
actuator_length=len(actuators)
model=models[model_type](input_dim=state_length+2*actuator_length, output_dim=state_length,
                         **model_hyperparams)
# dump to same location as the config filename, with .tar instead of .cfg
output_filename=os.path.join(config['model']['output_dir'],f"{config['model']['output_filename_base']}.tar")
# you probably want to use the same config file you had used for the original model, though you might swap
# out signals like for data+sim
if tune_model:
    untuned_output_filename=os.path.join(config['model']['output_dir'],f"{model_to_tune_filename_base}.tar")
    # note that if you run on a different computer, you might need map_location=torch.device('cpu') for loading
    saved_state=torch.load(untuned_output_filename)
    model.load_state_dict(saved_state['model_state_dict'])
    if resume_training:
        start_epoch=saved_state['epoch']+1
    print(f'Starting from model state stored in {untuned_output_filename}, from epoch {start_epoch}; saving new model to {output_filename}')
    for name, child in model.named_children():
        if name in frozen_layers:
            print(f"Freezing '{name}' layer for tuning procedure")
            for param in child.parameters():
                param.requires_grad = False

print('Organizing train data from preprocessed_data')
start_time=time.time()
x_train, y_train, shots, times = ian_dataset(preprocessed_data_filenamebase+'train.pkl',
                                             profiles, actuators, parameters,
                                             sort_by_size=True, min_sample_length=2*nwarmup)
print(f'...took {(time.time()-start_time):0.2f}s')
print('Organizing validation data from preprocessed_data')
start_time=time.time()
x_val, y_val, shots, times = ian_dataset(preprocessed_data_filenamebase+'val.pkl',
                                         profiles, actuators, parameters,
                                         sort_by_size=True, min_sample_length=2*nwarmup)
print(f'...took {(time.time()-start_time):0.2f}s')

# I divide out by myself since different sequences/batches have different sizes
loss_fn=torch.nn.MSELoss(reduction='sum')

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
param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()
size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))
start_time=time.time()
prev_time=start_time

train_x_buckets = make_bucket(x_train, bucket_size)
train_y_buckets = make_bucket(y_train, bucket_size)
train_length_buckets = [[len(arr) for arr in bucket] for bucket in train_x_buckets]

val_x_buckets = make_bucket(x_val, bucket_size)
val_y_buckets = make_bucket(y_val, bucket_size)
val_length_buckets = [[len(arr) for arr in bucket] for bucket in val_x_buckets]

# apply filter to handle case of freezing layers (happens above) for model tuning
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,30,50,70], gamma=lr_gamma, verbose=True)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_gamma, last_epoch=lr_stop_epoch)
if tune_model:
    if resume_training:
        optimizer.load_state_dict(saved_state['optimizer_state_dict'])
    avg_train_losses=saved_state['train_losses']
    avg_val_losses=saved_state['val_losses']
else:
    avg_train_losses=[]
    avg_val_losses=[]
for epoch in range(start_epoch, n_epochs):
    if autoregression_num_steps<=1 or epoch<=autoregression_start_epoch:
        reset_probability=1
    else:
        if epoch>autoregression_end_epoch:
            avg_steps=autoregression_num_steps
        else:
            y2=float(autoregression_num_steps)
            y1=float(1)
            x2=float(autoregression_end_epoch)
            x1=float(autoregression_start_epoch)
            avg_steps=(y2-y1)/(x2-x1) * (epoch-x1) + y1
        reset_probability=1./avg_steps
        print(f'Autoregression on, average timestep {avg_steps:0.1f}')
    model.train()
    train_losses=[]
    for which_bucket in torch.randperm(len(train_x_buckets)):
        random_order=torch.randperm(len(train_x_buckets[which_bucket]))
        x_bucket=[train_x_buckets[which_bucket][i] for i in random_order]
        y_bucket=[train_y_buckets[which_bucket][i] for i in random_order]
        length_bucket=[train_length_buckets[which_bucket][i] for i in random_order]

        padded_x=pad_sequence(x_bucket, batch_first=True)
        padded_y=pad_sequence(y_bucket, batch_first=True)
        padded_x=padded_x.to(device)
        padded_y=padded_y.to(device)

        optimizer.zero_grad()
        model_output=model(padded_x,reset_probability=reset_probability,nwarmup=nwarmup)
        train_loss=masked_loss(loss_fn,
                               model_output, padded_y,
                               length_bucket)
        # L1 regularization
        '''l1_reg = torch.tensor(0.0, device=device)
        for param in model.parameters():
            l1_reg += torch.abs(param).sum()
        train_loss += l1_lambda*l1_reg # lambda is the hyperparameter defined in cfg

        # L2 regularization
        l2_reg = torch.tensor(0.0, device=device)
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2).sum()
        train_loss += l2_lambda * l2_reg'''

        # Backpropagation
        train_loss.backward()
        optimizer.step()
        train_losses.append(train_loss.item())
    #scheduler.step()
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
            model_output = model(padded_x,reset_probability=reset_probability,nwarmup=nwarmup)
            val_loss = masked_loss(loss_fn,
                                   model_output, padded_y,
                                   length_bucket)
            val_losses.append(val_loss.item())
        avg_val_losses.append(sum(val_losses)/len(val_losses))
    print(f'{epoch+1:4d}/{n_epochs}({(time.time()-prev_time):0.2f}s)... train: {avg_train_losses[-1]:0.2e}, val: {avg_val_losses[-1]:0.2e};')
    if (not early_saving) or avg_val_losses[-1]==min(avg_val_losses):
        print(f"Checkpoint")
        torch.save({
            'epoch': epoch,
            'val_indices': val_indices,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': avg_train_losses,
            'val_losses': avg_val_losses,
            'profiles': profiles,
            'actuators': actuators,
            'parameters': parameters,
            'model_hyperparams': model_hyperparams,
            'exclude_ech': True
        }, output_filename)
    prev_time=time.time()

print(f'...took {(time.time()-start_time)/60:0.2f}min')
