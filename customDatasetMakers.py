import time
import h5py
import numpy as np
import pickle
import torch

import dataSettings

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
def allTimesInBounds(arr, cutoff):
    return np.all(np.abs(arr[~np.isnan(arr)])<cutoff)

def preprocess_data(processed_data_filename,
                    raw_data_filename,profiles,scalars,
                    shots=None,lookahead=1,
                    ip_minimum=None,ip_maximum=None,
                    excluded_runs=[],exclude_ech=True, max_num_shots=np.inf):
    print(f'Building dataset {processed_data_filename}...')
    start_time=time.time()
    with h5py.File(raw_data_filename,'r') as f:
        times=f['times'][:]
        processed_data={key: [] for key in profiles+scalars+['shotnum','times']}
        recorded_shots, recorded_times = [],[]
        available_shots = list(f.keys())
        available_shots.remove('times')
        available_shots.remove('spatial_coordinates')
        if shots is None:
            used_shots=available_shots
        else:
            used_shots=np.intersect1d(available_shots,[str(shot) for shot in shots])
        prev_time=time.time()
        included_shot_count,total_timestep_count,included_timestep_count = 0,0,0
        SHOTS_PER_PRINT = 1000
        for nshot,shot in enumerate(used_shots):
            keys_exist=False
            if np.all([key in f[shot].keys() for key in profiles+scalars]):
                normalized_dic=dataSettings.get_normalized_dic({key: f[shot][key][:] for key in profiles+scalars})
                keys_exist=True
            if keys_exist \
               and np.all([allTimesInBounds(normalized_dic[key],dataSettings.deviation_cutoff) for key in profiles+scalars]) \
               and not (exclude_ech and ('ech_pwr_total' in f[shot]) and np.sum(f[shot]['ech_pwr_total'][:])) \
               and not (('run_sql' in f[shot]) and (f[shot]['run_sql'][()].decode('utf-8') in excluded_runs)):
                shot_included=False
                for t_ind in range(len(times)-lookahead):
                    total_timestep_count+=1
                    ip_in_bounds=True
                    if (ip_minimum is not None) or (ip_maximum is not None):
                        if 'ip' not in f[shot].keys():
                            ip_in_bounds=False
                        else:
                            ip_window=f[shot]['ip'][t_ind:t_ind+lookahead+1]
                            if ip_minimum is not None:
                                ip_in_bounds=ip_in_bounds and np.all(ip_window>ip_minimum)
                            if ip_maximum is not None:
                                ip_in_bounds=ip_in_bounds and np.all(ip_window<ip_maximum)
                    if ip_in_bounds:
                        tmp_profiles_arr={}
                        tmp_scalars_arr={}
                        for profile in profiles:
                            tmp_profiles_arr[profile]=f[shot][profile][t_ind:t_ind+lookahead+1]
                        for scalar in scalars:
                            tmp_scalars_arr[scalar]=f[shot][scalar][t_ind:t_ind+lookahead+1]
                        if np.all([profiles_ok(tmp_profiles) for tmp_profiles in tmp_profiles_arr.values()]) \
                           and np.all([scalars_ok(tmp_scalars) for tmp_scalars in tmp_scalars_arr.values()]):
                            for profile in profiles:
                                processed_data[profile].append(tmp_profiles_arr[profile])
                            for scalar in scalars:
                                processed_data[scalar].append(tmp_scalars_arr[scalar])
                            processed_data['shotnum'].append(np.array([int(shot)]*(lookahead+1)))
                            processed_data['times'].append(times[t_ind:t_ind+lookahead+1])
                            included_timestep_count+=1
                            shot_included=True
                if shot_included:
                    included_shot_count+=1
            if not (nshot+1) % SHOTS_PER_PRINT:
                print(f'{(nshot+1):5d}/{len(used_shots)} shots ({(time.time()-prev_time):0.2e}s)')
                prev_time=time.time()
            if included_shot_count>=max_num_shots:
                print(f'Breaking early, max number of shots acquired ({max_num_shots})')
                break
    print(f'...took {(time.time()-start_time)/60:0.2f}min,',
          f'{included_shot_count}/{len(used_shots)} shots included,',
          f'{included_timestep_count}/{total_timestep_count} timesteps included')
    for signal in processed_data:
        processed_data[signal]=np.array(processed_data[signal])
    with open(processed_data_filename, 'wb') as f:
        pickle.dump(processed_data,f)

def ian_dataset(processed_data_filename,
                profiles, actuators, parameters,
                min_sample_length=6,
                sort_by_size=True):
    with open(processed_data_filename, 'rb') as f:
        processed_data=pickle.load(f)
    # normalize
    processed_data=dataSettings.get_normalized_dic(processed_data, excluded_sigs=['shotnum', 'times'])
    # for sig in profiles + parameters + actuators:
    #     processed_data[sig]=dataSettings.normalize(processed_data[sig], sig)
    in_sample,in_sample,out_sample,out_sample,shots,times=[],[],[],[],[],[]
    in_samples,out_samples=[],[]
    previous_processed_sample_ind, processed_sample_ind=0,0
    for processed_sample_ind in range(len(processed_data['times'])):
        in_timeslice=[]
        out_timeslice=[]
        # make each sample as long as possible while still being contiguous in time
        for profile in profiles:
            in_timeslice.extend(processed_data[profile][processed_sample_ind][0])
            out_timeslice.extend(processed_data[profile][processed_sample_ind][-1])
        for parameter in parameters:
            in_timeslice.append(processed_data[parameter][processed_sample_ind][0])
            out_timeslice.append(processed_data[parameter][processed_sample_ind][-1])
        for actuator in actuators:
            in_timeslice.append(processed_data[actuator][processed_sample_ind][0])
        for actuator in actuators:
            in_timeslice.append(processed_data[actuator][processed_sample_ind][-1])
        in_sample.append(in_timeslice)
        out_sample.append(out_timeslice)
        # move on to a new sample when we run out of contiguous chunks
        if processed_sample_ind==len(processed_data['times'])-1 \
           or processed_data['times'][processed_sample_ind][-1]!=processed_data['times'][processed_sample_ind+1][0] \
               or processed_data['shotnum'][processed_sample_ind][0]!=processed_data['shotnum'][processed_sample_ind+1][0]:
            if len(in_sample)>=min_sample_length:
                in_samples.append(torch.Tensor(in_sample))
                out_samples.append(torch.Tensor(out_sample))
                shots.append(processed_data['shotnum'][previous_processed_sample_ind][0])
                times.append(processed_data['times'][previous_processed_sample_ind][0])
            previous_processed_sample_ind=processed_sample_ind+1
            in_sample, out_sample=[],[]
    if sort_by_size:
        sample_lengths = [len(seq) for seq in in_samples]
        sorted_indices = sorted(range(len(sample_lengths)), key=sample_lengths.__getitem__, reverse=True)
        in_samples = [in_samples[i] for i in sorted_indices]
        out_samples = [out_samples[i] for i in sorted_indices]
        shots = [shots[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]
    return in_samples, out_samples, shots, times
