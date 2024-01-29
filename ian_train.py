import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from customDatasetMakers import preprocess_data, ian_dataset, get_state_indices_dic
from customModels import IanRNN, IanMLP, HiroLinear
from train_helpers import make_bucket, \
    get_state_mask, get_sample_time_state_mask, masked_loss

from dataSettings import nx

import configparser
import os
import sys
import shutil
import time

models={'IanRNN': IanRNN, 'IanMLP': IanMLP, 'HiroLinear': HiroLinear}

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename='model.cfg'

config=configparser.ConfigParser()
config.read(config_filename)
preprocessed_data_filenamebase=config['preprocess']['preprocessed_data_filenamebase']
use_fancy_normalization=config['preprocess'].getboolean('use_fancy_normalization',False)
model_type=config['model'].get('model_type','IanRNN')
bucket_size=config['optimization'].getint('bucket_size')
nwarmup=config['optimization'].getint('nwarmup',0)
n_epochs=config['optimization'].getint('n_epochs')
lr=config['optimization'].getfloat('lr')
lr_gamma=config['optimization'].getfloat('lr_gamma')
lr_stop_epoch=config['optimization'].getint('lr_stop_epoch')
early_saving=config['optimization'].getboolean('early_saving')
l1_lambda=config['optimization'].getfloat('l1_lambda')
l2_lambda=config['optimization'].getfloat('l2_lambda')
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs'].get('parameters','').split()
calculations=config['inputs'].get('calculations','').split()
save_epochs=config['optimization'].get('save_epochs','').split()
save_epochs=[int(elem) for elem in save_epochs]
autoregression_num_steps=config['optimization'].getfloat('autoregression_num_steps',1)
autoregression_start_epoch=config['optimization'].getint('autoregression_start_epoch',int(n_epochs/4))
autoregression_end_epoch=config['optimization'].getint('autoregression_end_epoch',int(3*n_epochs/4))
if autoregression_num_steps<1:
    autoregression_num_steps=1
# temporary to maintain back-compatibility
if config.has_section('tuning'):
    tune_model=config['tuning'].getboolean('tune_model',False)
    if tune_model:
        if 'model_to_tune_filename_base' not in config['tuning']:
            raise Exception("config['tuning']['tune_model'] set to true but no starting file specified in config['tuning']['model_to_tune_filename_base']")
        model_to_tune_filename_base=config['tuning']['model_to_tune_filename_base']
    frozen_layers=config['tuning'].get('frozen_layers','').split()
    resume_training=config['tuning'].getboolean('resume_training',False)
    masked_outputs=config['tuning'].get('masked_outputs','').split()
    rho_bdry_index=config['tuning'].get('rho_bdry_index',None)
else:
    tune_model=False
    masked_outputs=[]
    rho_bdry_index=None
# epoch to start on, should be 0 generally but can increase w/ tune_model to restart a model that stopped halfway
# at the moment, by default tune_model will start the epochs where the previous left off
start_epoch=0

model_hyperparams={key: int(val) for key,val in dict(config[model_type]).items()}

state_length=len(profiles)*nx+len(parameters)
actuator_length=len(actuators)
calculation_length=len(calculations)*33
model=models[model_type](input_dim=state_length+calculation_length+2*actuator_length, output_dim=state_length,
                         **model_hyperparams)
# dump to same location as the config filename, with .tar instead of .cfg
output_filename=os.path.join(config['model']['output_dir'],f"{config['model']['output_filename_base']}.tar")
epoch_output_filename = lambda epoch : os.path.join(config['model']['output_dir'],f"{config['model']['output_filename_base']}EPOCH{epoch}.tar")
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

min_sample_length=max(2*nwarmup,6)
train_filename=preprocessed_data_filenamebase+'train.pkl'
print(f'Organizing train data from {train_filename}')
start_time=time.time()
x_train, y_train, shots, times = ian_dataset(train_filename,
                                             profiles,parameters,calculations,actuators,
                                             sort_by_size=True, min_sample_length=min_sample_length,
                                             use_fancy_normalization=use_fancy_normalization)
print(f'...took {(time.time()-start_time):0.2f}s')
val_filename=preprocessed_data_filenamebase+'val.pkl'
print(f'Organizing validation data from {val_filename}')
start_time=time.time()
x_val, y_val, shots, times = ian_dataset(val_filename,
                                         profiles,parameters,calculations,actuators,
                                         sort_by_size=True, min_sample_length=min_sample_length,
                                         use_fancy_normalization=use_fancy_normalization)
print(f'...took {(time.time()-start_time):0.2f}s')

# I divide out by myself since different sequences/batches have different sizes
# see train_helpers.py
loss_fn=torch.nn.MSELoss(reduction='sum')
state_mask=get_state_mask(profiles, parameters,
                          masked_outputs, rho_bdry_index)
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
if tune_model and resume_training:
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
        model_output=model_output.to(device)
        mask=get_sample_time_state_mask(state_mask, model_output.size(), length_bucket, nwarmup)
        mask=mask.to(device)
        train_loss=masked_loss(loss_fn,
                               model_output, padded_y,
                               mask)
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
            model_output = model_output.to(device)
            mask=get_sample_time_state_mask(state_mask, model_output.size(), length_bucket, nwarmup)
            mask=mask.to(device)
            val_loss=masked_loss(loss_fn,
                                 model_output, padded_y,
                                 mask)
            val_losses.append(val_loss.item())
        avg_val_losses.append(sum(val_losses)/len(val_losses))
    print(f'{epoch+1:4d}/{n_epochs}({(time.time()-prev_time):0.2f}s)... train: {avg_train_losses[-1]:0.2e}, val: {avg_val_losses[-1]:0.2e};')
    # the task gets harder for curriculum learning during the ramp
    # before the ramp, consider only the best model so far
    if autoregression_num_steps<=1 or epoch<autoregression_start_epoch:
        relevant_val_losses=avg_val_losses
    else:
        # if during the ramp always save
        if epoch<=autoregression_end_epoch:
            relevant_val_losses=[avg_val_losses[-1]]
        # after ramp consider only losses after ramp
        else:
            # and if we're e.g. tuning a model on a different task only consider new loss regime
            relevant_val_losses=avg_val_losses[max(start_epoch,autoregression_end_epoch):]
    best_epoch= ( avg_val_losses[-1]==min(relevant_val_losses) )
    # in weird case we don't yet have a .tar file, e.g. if we're resuming training into a new filename,
    # be sure to save the first step
    if not os.path.exists(output_filename):
        best_epoch=True
    if (not early_saving) or best_epoch:
        print(f"Checkpoint")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            #'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': avg_train_losses,
            'val_losses': avg_val_losses,
            'profiles': profiles,
            'parameters': parameters,
            'calculations': calculations,
            'actuators': actuators,
            'model_hyperparams': model_hyperparams,
        }, output_filename)
    if epoch in save_epochs:
        shutil.copyfile(output_filename, epoch_output_filename)
    prev_time=time.time()

print(f'...took {(time.time()-start_time)/60:0.2f}min')
