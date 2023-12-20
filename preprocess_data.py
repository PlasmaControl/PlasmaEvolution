import customDatasetMakers

import configparser
import sys
import os

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename='preprocess.cfg'

config=configparser.ConfigParser()
config.read(config_filename)
raw_data_filename=config['logistics']['raw_data_filename']
preprocessed_data_filenamebase=os.path.join(config['logistics']['output_dir'],
                                            config['logistics']['output_filename_base'])
ip_minimum=config['settings'].getfloat('ip_minimum')
ip_maximum=config['settings'].getfloat('ip_maximum')
lookahead=config['settings'].getint('lookahead')
profiles=config['signals']['profiles_superset'].split()
scalars=config['signals']['scalars_superset'].split()
zero_fill_signals=config['settings'].get('zero_fill_signals','').split()
exclude_ech=config['settings'].getboolean('exclude_ech',True)
ech_threshold=config['settings'].getfloat('ech_threshold',0.1)
exclude_ich=config['settings'].getboolean('exclude_ich',True)
deviation_cutoff=config['settings'].getfloat('deviation_cutoff',10)

max_num_shots=config['shots'].getint('max_num_shots',200000) #small for testing
min_shot=config['shots'].getint('min_shot',0)
max_shot=config['shots'].getint('max_shot',200000)
val_index=config['shots'].getint('val_index',5)
test_index=config['shots'].getint('test_index',0)
excluded_runs=config['shots'].get('excluded_runs','').split()

datasetParams={'raw_data_filename': raw_data_filename, 'profiles': profiles, 'scalars': scalars,
               'lookahead': lookahead,
               'ip_minimum': ip_minimum, 'ip_maximum': ip_maximum,
               'zero_fill_signals': zero_fill_signals,
               'exclude_ech': exclude_ech, 'exclude_ich': exclude_ich,
               'ech_threshold': ech_threshold,
               'max_num_shots': max_num_shots,
               'deviation_cutoff': deviation_cutoff}

print(raw_data_filename)
train_shots=[shot for shot in range(min_shot,max_shot) if shot%10 not in [val_index,test_index]]
val_shots=[shot for shot in range(min_shot,max_shot) if shot%10 in [val_index]]
test_shots=[shot for shot in range(min_shot,max_shot) if shot%10 in [test_index]]
train_dataset=customDatasetMakers.preprocess_data(preprocessed_data_filenamebase+'train.pkl',shots=train_shots,**datasetParams)
val_dataset=customDatasetMakers.preprocess_data(preprocessed_data_filenamebase+'val.pkl',shots=val_shots,**datasetParams)
test_dataset=customDatasetMakers.preprocess_data(preprocessed_data_filenamebase+'test.pkl',shots=test_shots,**datasetParams)
# for ASTRA-TRANSP (or generally being careful about extrapolation) exclude the runs associated with shots you'll test on
if False:
    datasetParams['excluded_runs']=[]
# for testing individual shot_times (usually used as the test after training with excluded runs associated with these)
if False:
    shots=[175970, 175970]
    time_bounds=[[1000,1400], [2280,2680]]
    customDatasetMakers.preprocess_shot_times('small_test.pkl',
                                              shots=shots, time_bounds=time_bounds,
                                              **datasetParams)
