import torch
import time
import h5py
import numpy as np
from torch.utils.data import TensorDataset
from dataSettings import normalizations

def normalize(arr, sig_name):
    # q blows up at the edge, use iota = 1/q as proxy for q and don't use ad hoc normalization
    if 'qpsi' in sig_name:
        normed_arr = 1. / arr
    else:
        normed_arr = (arr - normalizations[sig_name]['mean']) / normalizations[sig_name]['std']
    return normed_arr
def denormalize(arr, sig_name):
    if 'qpsi' in sig_name:
        denormed_arr = 1. / arr
    else:
        denormed_arr = (arr * normalizations[sig_name]['std']) + normalizations[sig_name]['mean']
    return denormed_arr

def standard_dataset(data_filename,profiles,actuators,parameters,lookahead,lookback,
                     latest_output_only=False, exclude_ech=True,
                     shots=None, excluded_runs=[], extra_sigs=['shots', 'times']):
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
        input_profiles, input_actuators, input_parameters, output_profiles = [],[],[],[]
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
                    tmp_input_profiles=[]
                    for profile in profiles:
                        tmp_input_profiles.append(f[shot][profile][t_ind])
                    tmp_output_profiles=[]
                    for profile in profiles:
                        if latest_output_only:
                            tmp_output_profiles.append(f[shot][profile][t_ind+lookahead])
                        else:
                            tmp_output_profiles.append(f[shot][profile][t_ind+1:t_ind+lookahead+1])
                    tmp_input_actuators=[]
                    for actuator in actuators:
                        tmp_input_actuators.append(f[shot][actuator][t_ind-lookback:t_ind+lookahead+1])
                    tmp_input_parameters=[]
                    for parameter in parameters:
                        tmp_input_parameters.append(f[shot][parameter][t_ind-lookback:t_ind+1])
                    tmp_input_profiles=np.array(tmp_input_profiles)
                    tmp_output_profiles=np.array(tmp_output_profiles)
                    tmp_input_actuators=np.array(tmp_input_actuators)
                    tmp_input_parameters=np.array(tmp_input_parameters)
                    if profiles_ok(tmp_input_profiles) and profiles_ok(tmp_output_profiles) \
                       and scalars_ok(tmp_input_actuators) and scalars_ok(tmp_input_parameters):
                        input_profiles.append(tmp_input_profiles)
                        input_actuators.append(tmp_input_actuators)
                        input_parameters.append(tmp_input_parameters)
                        output_profiles.append(tmp_output_profiles)
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
    input_profiles_tensor=torch.as_tensor(np.array(input_profiles)).float()
    output_profiles_tensor=torch.as_tensor(np.array(output_profiles)).float()
    input_actuators_tensor=torch.as_tensor(np.array(input_actuators)).float()
    input_parameters_tensor=torch.as_tensor(np.array(input_parameters)).float()
    recorded_shots_tensor=torch.as_tensor(np.array(recorded_shots)).int()
    recorded_times_tensor=torch.as_tensor(np.array(recorded_times)).int()

    for i,profile in enumerate(profiles):
        input_profiles_tensor[:,i] = normalize(input_profiles_tensor[:,i], profile)
        output_profiles_tensor[:,i] = normalize(output_profiles_tensor[:,i], profile)
    for i,actuator in enumerate(actuators):
        input_actuators_tensor[:,i] = normalize(input_actuators_tensor[:,i], actuator)
    for i,parameter in enumerate(parameters):
        input_parameters_tensor[:,i] = normalize(input_parameters_tensor[:,i], parameter)

    # reshape to be in format that pytorch likes for RNN stuff
    if not latest_output_only:
        output_profiles_tensor = output_profiles_tensor.transpose(-3,-2)
    input_actuators_tensor = input_actuators_tensor.transpose(-2,-1)
    input_parameters_tensor = input_parameters_tensor.transpose(-2,-1)

    extra_sig_map={
        'shots': recorded_shots_tensor,
        'times': recorded_times_tensor
        }
    extra_sigs_tensor=torch.stack([extra_sig_map[key] for key in extra_sigs]).T

    return TensorDataset(output_profiles_tensor, input_profiles_tensor,
                         input_actuators_tensor, input_parameters_tensor,
                         extra_sigs_tensor)
