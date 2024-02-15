import os
import configparser
import shutil

def launch_preprocess(baseconfig_filename='preprocess.cfg',submit_runs=False,hyperparam_adjustments=[{}]):
    root_dir=os.path.dirname(os.path.realpath(__file__))
    config=configparser.ConfigParser()
    config.read(baseconfig_filename)
    for hyperparam_adjustment in hyperparam_adjustments:
        for category in hyperparam_adjustment:
            for hyperparam in hyperparam_adjustment[category]:
                config[category][hyperparam]=str(hyperparam_adjustment[category][hyperparam])
        output_dir=config['logistics']['output_dir']
        output_filename_base=config['logistics']['output_filename_base']

        config_filename=os.path.join(output_dir,f'{output_filename_base}config')
        #shutil.copyfile(baseconfig_filename, config_filename)
        with open(config_filename,'w') as f:
           config.write(f)
        log_filename=os.path.join(output_dir,f'{output_filename_base}log.out')
        slurm_text=f'''#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem 48G
#SBATCH -o {log_filename}
#SBATCH -t 2-00:00:00

root_dir={root_dir}
module load anaconda
conda activate torch
cd $root_dir
python -u preprocess_data.py {config_filename}

exit'''
        slurm_filename=os.path.join(output_dir,f'{output_filename_base}job.slurm')
        with open(slurm_filename,'w') as f:
            f.write(slurm_text)
        if submit_runs:
            os.system(f'sbatch {slurm_filename}')

if __name__=='__main__':
    # increase n_models to do ensembling, submit_batch=False for testing in the function call below
    # empty array means use the base config file
    hyperparam_adjustments=[{}]
    # an example of how to make hyperparameter adjustments for training models with different inputs across different training sets
    # (standard workflow for data+sim paper)
    if True:
        hyperparam_adjustments=[]
        # dealing with training set
        preprocessed_filename_dic={'all': {'ip_minimum': 0.0e6, 'ip_maximum': 10.0e6},
                                   'ip_0_900': {'ip_minimum': 0.0e6, 'ip_maximum': 0.9e6},
                                   'ip_0_1200': {'ip_minimum': 0.0e6, 'ip_maximum': 1.2e6}}
        # dealing with inputs
        for which_trainset in preprocessed_filename_dic:
            hyperparam_adjustments.append(
                {
                    'logistics': {'output_filename_base': f'astraInterpretiveAndTGLFNN{which_trainset}'},
                    'settings': preprocessed_filename_dic[which_trainset]
                }
            )
    launch_preprocess(baseconfig_filename='preprocess.cfg',submit_runs=True,hyperparam_adjustments=hyperparam_adjustments)
