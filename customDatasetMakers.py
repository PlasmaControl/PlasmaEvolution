import torch
import time
import h5py
import numpy as np
from torch.utils.data import TensorDataset

import dataSettings

def standard_dataset(data_filename,profiles,actuators,parameters,lookahead,lookback,
                     shots=None, excluded_runs=[],
                     exclude_ech=True, profile_lookback=1, extra_sigs=['shots', 'times']):
    def profiles_ok(profiles):
        if np.isnan(profiles).any():
            return False
        # also remove if profile is all 0 spatially
        if not np.sum(np.abs(profiles),axis=-1).all():
            return False
        return True
    def scalars_ok(scalars):
        if np.isnan(scalars).any():
            return False
        return True

    print('Building dataset...')
    start_time=time.time()
    with h5py.File(data_filename,'r') as f:
        times=f['times'][:]
        profiles_arr, actuators_arr, parameters_arr = [],[],[]
        recorded_shots, recorded_times = [],[]
        if shots is None:
            shots = list(f.keys())
            shots.remove('times')
            shots.remove('spatial_coordinates')
        else:
            shots=[str(shot) for shot in shots]
        prev_time=time.time()
        included_shot_count,total_timestep_count,included_timestep_count = 0,0,0
        SHOTS_PER_PRINT = 200
        for nshot,shot in enumerate(shots):
            if (shot in f) and np.all([key in f[shot].keys() for key in actuators+profiles+parameters]) \
               and not (exclude_ech and ('ech_pwr_total' in f[shot]) and np.sum(f[shot]['ech_pwr_total'][:])) \
               and not (f[shot]['run_sql'][()].decode('utf-8') in excluded_runs):
                shot_included=False
                for t_ind in range(lookback,len(times)-lookahead):
                    total_timestep_count+=1
                    tmp_profiles_arr=[]
                    for profile in profiles:
                        tmp_profiles_arr.append(f[shot][profile][t_ind-profile_lookback:t_ind+lookahead+1])
                    tmp_actuators_arr=[]
                    for actuator in actuators:
                        tmp_actuators_arr.append(f[shot][actuator][t_ind-lookback:t_ind+lookahead+1])
                    tmp_parameters_arr=[]
                    for parameter in parameters:
                        tmp_parameters_arr.append(f[shot][parameter][t_ind-lookback:t_ind+1])
                    tmp_profiles_arr=np.array(tmp_profiles_arr)
                    tmp_actuators_arr=np.array(tmp_actuators_arr)
                    tmp_parameters_arr=np.array(tmp_parameters_arr)
                    if profiles_ok(tmp_profiles_arr) \
                       and scalars_ok(tmp_actuators_arr) and scalars_ok(tmp_parameters_arr):
                        profiles_arr.append(tmp_profiles_arr)
                        actuators_arr.append(tmp_actuators_arr)
                        parameters_arr.append(tmp_parameters_arr)
                        recorded_shots.append(int(shot))
                        recorded_times.append(times[t_ind])
                        included_timestep_count+=1
                        shot_included=True
                if shot_included:
                    included_shot_count+=1
            if (nshot+1) % SHOTS_PER_PRINT:
                print(f'{(nshot+1):5d}/{len(shots)} shots ({(time.time()-prev_time)/SHOTS_PER_PRINT:0.2f}s/shot)')
                prev_time=time.time()
    print(f'...took {(time.time()-start_time)/60:0.2f}min,',
          f'{included_shot_count}/{len(shots)} shots included,',
          f'{included_timestep_count}/{total_timestep_count} timesteps included')
    profiles_tensor=torch.as_tensor(np.array(profiles_arr)).float()
    actuators_tensor=torch.as_tensor(np.array(actuators_arr)).float()
    parameters_tensor=torch.as_tensor(np.array(parameters_arr)).float()
    recorded_shots_tensor=torch.as_tensor(np.array(recorded_shots)).int()
    recorded_times_tensor=torch.as_tensor(np.array(recorded_times)).int()

    for i,profile in enumerate(profiles):
        profiles_tensor[:,i] = dataSettings.normalize(profiles_tensor[:,i], profile)
    for i,actuator in enumerate(actuators):
        actuators_tensor[:,i] = dataSettings.normalize(actuators_tensor[:,i], actuator)
    for i,parameter in enumerate(parameters):
        parameters_tensor[:,i] = dataSettings.normalize(parameters_tensor[:,i], parameter)
    # reshape toward format that pytorch likes for RNN stuff when batch_first=True:
    # scalars: N x L x Nscalars for N batch, L sequence length (lookback+1; lookback+1 + lookahead)
    # profiles: N x L x Nprofiles x NrhoPoints for N batch, L sequence length (lookback+1; lookahead)
    #           where for profiles we need to flatten/unflatten the Nprofiles x NrhoPoints in the model
    profiles_tensor = profiles_tensor.transpose(-3,-2)
    actuators_tensor = actuators_tensor.transpose(-2,-1)
    parameters_tensor = parameters_tensor.transpose(-2,-1)

    extra_sig_map={
        'shots': recorded_shots_tensor,
        'times': recorded_times_tensor
    }
    extra_sigs_tensor=torch.stack([extra_sig_map[key] for key in extra_sigs]).T

    return TensorDataset(profiles_tensor, actuators_tensor, parameters_tensor,
                         extra_sigs_tensor)
