import pickle
import json
import time
import configparser
from customDatasetMakers import ian_dataset
import dataSettings
import sys
import pdb
import random

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename='model.cfg'

config=configparser.ConfigParser()
config.read(config_filename)
which_dataset='all'
preprocessed_data_filenamebase=f'/projects/EKOLEMEN/profile_predictor/final_paper/{which_dataset}' #config['preprocess']['preprocessed_data_filenamebase']
profiles=config['inputs']['profiles'].split()
actuators=config['inputs']['actuators'].split()
parameters=config['inputs'].get('parameters','').split()
calculations=config['inputs'].get('calculations','').split()
nwarmup=config['optimization'].getint('nwarmup',0)
min_sample_length=25 #max(2*nwarmup,20)
json_stuff=[]
for dataset in ['test','val','train']: #['test']: #['val','test','train']:
    preprocessed_filename=preprocessed_data_filenamebase+dataset+'.pkl'
    profiling_time=time.time()
    print(f'Gathering shot/times for {preprocessed_filename}')
    x, y, shots, times = ian_dataset(preprocessed_filename,
                                     profiles,parameters,calculations,actuators,
                                     sort_by_size=True, min_sample_length=min_sample_length)
    print(f'...took {(time.time()-profiling_time):0.2f}s')
    for ind in range(len(shots)):
        info={}
        info['shot']=int(shots[ind])
        start_time=int(times[ind])/1.e3
        num_times=len(x[ind])
        info['start_time']=round(start_time,2)
        info['end_time']=round(start_time+num_times*dataSettings.DT,2)
        info['astra_submitted']=False
        json_stuff.append(info)
json_filename=f'{which_dataset}_test.json'
print(f'Dumping to {json_filename}')
random.shuffle(json_stuff)
with open(json_filename,'w') as f:
    json.dump(json_stuff, f, indent=2)

# import pickle
# import json

# json_filename='ip_1000_1200_val.json'
# pickle_filename='/scratch/gpfs/jabbate/paper_results/rollout_ip_0_1200withSim_epoch500_5steps_ip_1000_1200val.pkl'

# with open(pickle_filename,'rb') as f:
#     data=pickle.load(f)
# keys=list(data.keys())
# json_stuff=[]
# # by default these will be from longest samples to shortest samples
# # note there can be duplicate shots
# for key in keys:
#     info={}
#     split_string=key.split('_')
#     info['shot']=int(split_string[0])
#     info['start_time']=int(split_string[1])/1000.
#     info['end_time']=int(split_string[2])/1000.
#     info['astra_submitted']=False
#     json_stuff.append(info)
# with open(json_filename,'w') as f:
#     json.dump(json_stuff, f, indent=2)
