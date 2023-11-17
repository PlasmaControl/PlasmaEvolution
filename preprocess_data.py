import customDatasetMakers

import configparser
import sys

from dataSettings import nx, train_shots, val_shots, test_shots, val_indices

if (len(sys.argv)-1) > 0:
    config_filename=sys.argv[1]
else:
    config_filename='preprocess.cfg'

config=configparser.ConfigParser()
config.read(config_filename)
raw_data_filename=config['preprocess']['raw_data_filename']
preprocessed_data_filenamebase=config['preprocess']['preprocessed_data_filenamebase']
ip_minimum=config['preprocess'].getfloat('ip_minimum')
ip_maximum=config['preprocess'].getfloat('ip_maximum')
lookahead=config['preprocess'].getint('lookahead')
profiles=config['preprocess']['profiles_superset'].split()
scalars=config['preprocess']['scalars_superset'].split()
zero_fill_signals=config['preprocess'].get('zero_fill_signals','').split()
exclude_ech=config['preprocess'].getboolean('exclude_ech',True)

datasetParams={'raw_data_filename': raw_data_filename, 'profiles': profiles, 'scalars': scalars,
               'lookahead': lookahead,
               'ip_minimum': ip_minimum, 'ip_maximum': ip_maximum,
               'zero_fill_signals': zero_fill_signals, 'exclude_ech': exclude_ech}

# useful for testing
if False:
    datasetParams['max_num_shots']=2

print(raw_data_filename)
train_dataset=customDatasetMakers.preprocess_data(preprocessed_data_filenamebase+'train.pkl',shots=train_shots,**datasetParams)
val_dataset=customDatasetMakers.preprocess_data(preprocessed_data_filenamebase+'val.pkl',shots=val_shots,**datasetParams)
test_dataset=customDatasetMakers.preprocess_data(preprocessed_data_filenamebase+'test.pkl',shots=test_shots,**datasetParams)
# for ASTRA-TRANSP (or generally being careful about extrapolation) exclude the runs associated with shots you'll test on
if False:
    datasetParams['excluded_runs']=['20190628B', '20130911', '20100317', '20171012', '20220719', '20200924', '20150107',
                                    '20190821', '20100325A', '20190620B', '20160119', '20130417A', '20220623', '20180227A',
                                    '20180201', '20160204A', '20120713A', '20210416B', '20180327A', '20190729', '20210315A',
                                    '20200929A', '20220628', '20111201', '20130625', '20180124', '20110526', '20190822',
                                    '20180118', '20160331A', '20160108', '20171011', '20110818', '20170413', '20190812A',
                                    '20150105', '20170310', '20190816A', '20151215', '20160127', '20220825', '20220707',
                                    '20210224', '20210429A', '20220810', '20130912', '20210204A', '20220722A', '20210301',
                                    '20170713', '20171010', '20131003', '20120813', '20140513', '20160128A', '20180131',
                                    '20120622', '20110811', '20191107A', '20150127A', '20190808A', '20131011', '20171221A',
                                    '20210126A', '20141111', '20111108B', '20170804', '20140728', '20210416A', '20220628A',
                                    '20130820', '20131016', '20170720', '20220518', '20210615', '20150724', '20200619',
                                    '20160404A', '20120731', '20140626']
# for testing individual shot_times (usually used as the test after training with excluded runs associated with these)
if False:
    shots=[175970, 175970]
    time_bounds=[[1000,1400], [2280,2680]]
    customDatasetMakers.preprocess_shot_times('small_test.pkl',
                                              shots=shots, time_bounds=time_bounds,
                                              **datasetParams)
