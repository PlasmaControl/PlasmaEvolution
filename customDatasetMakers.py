import time
import h5py
import numpy as np
import pickle
import torch

import dataSettings

verbose=False

def profiles_ok(profiles, remove_all_zero_profiles=True):
    if np.isnan(profiles).any():
        return False
    # also remove if profile is all 0 spatially
    if remove_all_zero_profiles:
        if not np.sum(np.abs(profiles),axis=-1).all():
            return False
    return True
def scalars_ok(scalars):
    if np.isnan(scalars).any():
        return False
    return True
def allTimesInBounds(arr, cutoff):
    return np.all(np.abs(arr[~np.isnan(arr)])<cutoff)
def check_signal_off(signal, threshold=0.1):
    return (np.all(np.isnan(signal)) or np.nanmax(signal)<threshold)

# to get excluded_runs for list of shots, run the following in OMFIT:
#
# query="""SELECT summaries.shot,shots.run,runs.brief
#          FROM summaries
#          LEFT JOIN shots ON summaries.shot=shots.shot
#          LEFT JOIN runs ON runs.run=shots.run
#          WHERE summaries.shot in {}
#       """.format(
#     '({})'.format(','.join([str(elem) for elem in shots]))
#     )
# sql=OMFITrdb(query,db='d3drdb',server='d3drdb',by_column=True)
# runs=list(set(sql['run']))
# print(str(runs))

# also note zero_fill_signals won't have outliers excluded
def preprocess_data(processed_data_filename,
                    raw_data_filename,profiles,scalars,
                    shots=None,lookahead=1,
                    ip_minimum=None,ip_maximum=None,
                    excluded_runs=[],exclude_ech=True,ech_threshold=0.1,
                    exclude_ich=True,
                    max_num_shots=np.inf,
                    zero_fill_signals=[]):
    if processed_data_filename is not None:
        print(f'Building dataset {processed_data_filename}...')
    else:
        print(f'Building dataset to return (not to dump to file)')
    start_time=time.time()
    # the below would be a bug sort of, want to deal with each profile individually
    remove_all_zero_profiles=True #not any([profile in zero_fill_signals for profile in profiles])
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
        if verbose:
            print(used_shots)
        prev_time=time.time()
        included_shot_count,total_timestep_count,included_timestep_count = 0,0,0
        SHOTS_PER_PRINT = 1000
        for nshot,shot in enumerate(used_shots):
            if verbose:
                print(shot)
            keys_exist=False
            needed_signals=profiles+scalars
            for sig in zero_fill_signals:
                if sig in needed_signals:
                    needed_signals.remove(sig)
            if np.all([key in f[shot].keys() for key in needed_signals]):
                normalized_dic=dataSettings.get_normalized_dic({key: f[shot][key][:] for key in needed_signals})
                keys_exist=True
            if keys_exist:
                within_deviation=True
                for signal in needed_signals:
                    if signal not in dataSettings.clipped_signals:
                        if not allTimesInBounds(normalized_dic[signal],dataSettings.deviation_cutoff):
                            within_deviation=False
                ech_ok=not (exclude_ech and ('ech_pwr_total' in f[shot]) and not check_signal_off(f[shot]['ech_pwr_total'][:], threshold=ech_threshold))
                ich_ok=not (exclude_ich and ('ich_pwr_total' in f[shot]) and not check_signal_off(f[shot]['ich_pwr_total'][:], threshold=0.1))
                run_ok=not (('run_sql' in f[shot]) and (f[shot]['run_sql'][()].decode('utf-8') in excluded_runs))
                if verbose:
                    if not within_deviation:
                        print(f'not within deviation_cutoff')
                        for key in needed_signals:
                            if not allTimesInBounds(normalized_dic[key],dataSettings.deviation_cutoff):
                                print(key)
                    if not ech_ok:
                        print(f"ech sum: {np.sum(f[shot]['ech_pwr_total'][:])}")
                    if not ich_ok:
                        print(f"ich sum: {np.sum(f[shot]['ich_pwr_total'][:])}")
                    if not run_ok:
                        print(f'run in excluded_runs')
            elif verbose:
                print('missing key(s):')
                for key in profiles+scalars:
                    if not key in f[shot].keys():
                        print(key)
            if keys_exist \
               and within_deviation \
               and ech_ok \
               and run_ok:
                shot_included=False
                for t_ind in range(len(times)-lookahead):
                    total_timestep_count+=1
                    ip_in_bounds=True
                    if (ip_minimum is not None) or (ip_maximum is not None):
                        if 'ip' not in f[shot].keys():
                            ip_in_bounds=False
                            if verbose:
                                print('ip not in file')
                        else:
                            ip_window=f[shot]['ip'][t_ind:t_ind+lookahead+1]
                            if ip_minimum is not None:
                                ip_in_bounds=ip_in_bounds and np.all(ip_window>ip_minimum)
                            if ip_maximum is not None:
                                ip_in_bounds=ip_in_bounds and np.all(ip_window<ip_maximum)
                            if not ip_in_bounds and verbose:
                                print(f'ip out of bounds for timestep ({ip_window})')
                    if ip_in_bounds:
                        tmp_profiles_arr={}
                        tmp_scalars_arr={}
                        for profile in profiles:
                            if (profile in zero_fill_signals) and (profile not in f[shot]):
                                tmp_profiles_arr[profile]=np.zeros(lookahead+1,dataSettings.nx)
                            else:
                                tmp_profiles_arr[profile]=f[shot][profile][t_ind:t_ind+lookahead+1]
                        for scalar in scalars:
                            if (scalar in zero_fill_signals) and (scalar not in f[shot]):
                                tmp_scalars_arr[scalar]=np.zeros(lookahead+1)
                            else:
                                tmp_scalars_arr[scalar]=f[shot][scalar][t_ind:t_ind+lookahead+1]
                        if np.all([profiles_ok(tmp_profiles, remove_all_zero_profiles) for tmp_profiles in tmp_profiles_arr.values()]) \
                           and np.all([scalars_ok(tmp_scalars) for tmp_scalars in tmp_scalars_arr.values()]):
                            for profile in profiles:
                                processed_data[profile].append(tmp_profiles_arr[profile])
                            for scalar in scalars:
                                processed_data[scalar].append(tmp_scalars_arr[scalar])
                            processed_data['shotnum'].append(np.array([int(shot)]*(lookahead+1)))
                            processed_data['times'].append(times[t_ind:t_ind+lookahead+1])
                            included_timestep_count+=1
                            shot_included=True
                        elif verbose:
                            print('not all profiles and scalars ok for timestep')
                            for profile in profiles:
                                if not profiles_ok(tmp_profiles_arr[profile]):
                                    print(f"{profile}: {tmp_profiles_arr[profile]}")
                            for scalar in scalars:
                                if not scalars_ok(tmp_scalars_arr[scalar]):
                                    print(f"{scalar}: {tmp_scalars_arr[scalar]}")
                            print(f"{profiles_ok(tmp_profiles) for tmp_profiles in tmp_profiles_arr.values()}")
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
        if signal in dataSettings.clipped_signals:
            processed_data[signal]=np.clip(processed_data[signal],
                                           dataSettings.clipped_signals[signal]['min'],
                                           dataSettings.clipped_signals[signal]['max'])
    if processed_data_filename is not None:
        with open(processed_data_filename, 'wb') as f:
            pickle.dump(processed_data,f)
    else:
        return processed_data

# lets you included nan for autoregressive predictions
# shots is a list of shots; initial_times is a list of start times (in ms);
# end_times is a list of end times (in ms)
def preprocess_shot_times(processed_data_filename,
                          raw_data_filename,profiles,scalars,
                          shots, time_bounds,
                          lookahead=1,
                          ip_minimum=None,ip_maximum=None,
                          excluded_runs=[],exclude_ech=True, exclude_ich=True,
                          max_num_shots=np.inf):
    print(f'Building dataset {processed_data_filename}...')
    start_time=time.time()
    with h5py.File(raw_data_filename,'r') as f:
        times=f['times'][:]
        processed_data={key: [] for key in profiles+scalars+['shotnum','times']}
        recorded_shots, recorded_times = [],[]
        available_shots = list(f.keys())
        available_shots.remove('times')
        available_shots.remove('spatial_coordinates')
        prev_time=time.time()
        included_shot_count,total_timestep_count,included_timestep_count = 0,0,0
        SHOTS_PER_PRINT = 1000
        shots=[str(shot) for shot in shots]
        for nshot,shot in enumerate(shots):
            if shot not in available_shots:
                print('shot not in {raw_data_filename}, going to next shot')
                continue
            initial_time=time_bounds[nshot][0]
            end_time=time_bounds[nshot][1]
            start_ind=np.argmin(np.abs(times-initial_time))
            end_ind=np.argmin(np.abs(times-end_time))
            print(shot)
            keys_exist=False
            if np.all([key in f[shot].keys() for key in profiles+scalars]):
                normalized_dic=dataSettings.get_normalized_dic({key: f[shot][key][:] for key in profiles+scalars})
                keys_exist=True
            if keys_exist:
                within_deviation=True
                for signal in profiles+scalars:
                    if signal not in dataSettings.clipped_signals:
                        if not allTimesInBounds(normalized_dic[signal],dataSettings.deviation_cutoff):
                            within_deviation=False
                ech_ok=not (('ech_pwr_total' in f[shot]) and not check_signal_off(f[shot]['ech_pwr_total'][:], threshold=0.1))
                ich_ok=not (('ich_pwr_total' in f[shot]) and not check_signal_off(f[shot]['ich_pwr_total'][:], threshold=0.1))
                if verbose:
                    if not within_deviation:
                        print(f'not within deviation_cutoff')
                    if not ech_ok:
                        print(f"ech sum: {np.sum(f[shot]['ech_pwr_total'][:])}")
                    if not ich_ok:
                        print(f"ich sum: {np.sum(f[shot]['ich_pwr_total'][:])}")
            elif verbose:
                print('missing key(s):')
            if not np.all([profiles_ok(f[shot][profile][start_ind]) for profile in profiles]):
                print(f"{shot} bad profiles, going to next shot")
                continue
            if keys_exist:
                for t_ind in range(start_ind, end_ind+1):
                    tmp_profiles_arr={}
                    tmp_scalars_arr={}
                    for profile in profiles:
                        tmp_profiles_arr[profile]=f[shot][profile][t_ind:t_ind+lookahead+1]
                    for scalar in scalars:
                        tmp_scalars_arr[scalar]=f[shot][scalar][t_ind:t_ind+lookahead+1]
                    if np.all([scalars_ok(tmp_scalars) for tmp_scalars in tmp_scalars_arr.values()]):
                        for profile in profiles:
                            processed_data[profile].append(tmp_profiles_arr[profile])
                        for scalar in scalars:
                            processed_data[scalar].append(tmp_scalars_arr[scalar])
                        processed_data['shotnum'].append(np.array([int(shot)]*(lookahead+1)))
                        processed_data['times'].append(times[t_ind:t_ind+lookahead+1])
                        included_timestep_count+=1
                    else:
                        print(f'Shot {shot} stopping at {times[t_ind]} (was supposed to be {times[start_ind]}-{times[end_ind]})')
                        break
            if not (nshot+1) % SHOTS_PER_PRINT:
                print(f'{(nshot+1):5d}/{len(shots)} shots ({(time.time()-prev_time):0.2e}s)')
                prev_time=time.time()
    print(f'...took {(time.time()-start_time)/60:0.2f}min,')
    for signal in processed_data:
        processed_data[signal]=np.array(processed_data[signal])
    with open(processed_data_filename, 'wb') as f:
        pickle.dump(processed_data,f)

def ian_dataset(processed_data_filename,
                profiles,parameters=[],calculations=[],actuators=[],
                min_sample_length=6,
                sort_by_size=True):
    with open(processed_data_filename, 'rb') as f:
        processed_data=pickle.load(f)
    # normalize
    processed_data=dataSettings.get_normalized_dic(processed_data, excluded_sigs=['shotnum', 'times'])
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
        for calculation in calculations:
            in_timeslice.extend(processed_data[calculation][processed_sample_ind][0])
        # in future can have this be a loop over lookahead
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

# made to be consistent with ian_dataset, double check it matches the above
# returns a dictionary corresponding to the indices occupied by each signal
def get_state_indices_dic(profiles, parameters, calculations=[], actuators=[], nx=dataSettings.nx, lookahead=1):
    indices_dic={actuator: [] for actuator in actuators}
    ind,next_ind=0,0
    for profile in profiles:
        next_ind=ind+nx
        indices_dic[profile]=list(range(ind,next_ind))
        ind=next_ind
    for sig in parameters:
        indices_dic[sig]=ind
        ind=ind+1
    for calculation in calculations:
        next_ind=ind+nx
        indices_dic[calculation]=list(range(ind,next_ind))
        ind=next_ind
    for lookahead in range(lookahead+1):
        for sig in actuators:
            indices_dic[sig].append(ind)
            ind=ind+1
    return indices_dic

# actuators is [] since the output state only has profiles and parameters,
# but the input state has actuators at t and t+1 also
# if only one state, wrap it like state_arrs=[state_arr] to call this fxn
def state_to_dic(state_arrs, profiles, parameters, calculations=[], actuators=[], nx=dataSettings.nx):
    indices_dic=get_state_indices_dic(profiles, parameters, calculations, actuators, nx=nx)
    state_arrs=np.array(state_arrs)
    num_states=len(state_arrs)
    dic={}
    for sig in profiles+parameters+calculations+actuators:
        dic[sig]=state_arrs[...,indices_dic[sig]]
    return dic

def dic_to_state(dic, profiles, parameters, calculations=[], actuators=[], nx=dataSettings.nx):
    for sig in dic:
        dic[sig]=torch.tensor(dic[sig]).to(torch.float)
    state_length=nx*len(profiles)+len(parameters)+nx*len(calculations)+len(actuators)*2
    dims=dic[profiles[0]].size()
    if len(dims)>1:
        num_states=dims[0]
        state_arrs=torch.zeros((num_states,state_length))
    else:
        state_arrs=torch.zeros(state_length)
    indices_dic=get_state_indices_dic(profiles, parameters, calculations, actuators, nx=nx)
    for sig in indices_dic:
        state_arrs[...,indices_dic[sig]]=dic[sig]
    return state_arrs
