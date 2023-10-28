import customDatasetMakers

import configparser
import sys

from dataSettings import nx, train_shots, val_shots, test_shots, val_indices

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename='configs/default.cfg'

config=configparser.ConfigParser()
config.read(config_filename)
raw_data_filename=config['preprocess']['raw_data_filename']
preprocessed_data_filenamebase=config['preprocess']['preprocessed_data_filenamebase']
ip_minimum=float(config['preprocess']['ip_minimum'])
ip_maximum=float(config['preprocess']['ip_maximum'])
lookahead=int(config['preprocess']['lookahead'])
profiles=config['preprocess']['profiles_superset'].split()
scalars=config['preprocess']['scalars_superset'].split()

datasetParams={'raw_data_filename': raw_data_filename, 'profiles': profiles, 'scalars': scalars,
               'lookahead': lookahead,
               'ip_minimum': ip_minimum, 'ip_maximum': ip_maximum}

# useful for testing
if False:
    datasetParams['max_num_shots']=2

print(raw_data_filename)
train_dataset=customDatasetMakers.preprocess_data(preprocessed_data_filenamebase+'train.pkl',shots=train_shots,**datasetParams)
val_dataset=customDatasetMakers.preprocess_data(preprocessed_data_filenamebase+'val.pkl',shots=val_shots,**datasetParams)
test_dataset=customDatasetMakers.preprocess_data(preprocessed_data_filenamebase+'test.pkl',shots=test_shots,**datasetParams)
