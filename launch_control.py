import os
import configparser

def launch_control(baseconfig_filename='model.cfg',submit_runs=False, hyperparam_adjustments=[{}]):
    root_dir=os.path.dirname(os.path.realpath(__file__))
    config=configparser.ConfigParser()
    config.read(baseconfig_filename)
    for hyperparam_adjustment in hyperparam_adjustments:
        for category in hyperparam_adjustment:
            for hyperparam in hyperparam_adjustment[category]:
                config[category][hyperparam]=str(hyperparam_adjustment[category][hyperparam])
        output_dir=config['model']['output_dir']
        output_filename_base=config['model']['output_filename_base']

        config_filename=os.path.join(output_dir,f'control_{output_filename_base}config')
        #shutil.copyfile(baseconfig_filename, config_filename)
        with open(config_filename,'w') as f:
           config.write(f)
        log_filename=os.path.join(output_dir,f'control_{output_filename_base}log.out')
        slurm_text=f'''#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem 1G
#SBATCH -o {log_filename}
#SBATCH -t 00:55:00

root_dir={root_dir}
module load anaconda3/2022.5
conda activate torch
cd $root_dir
python -u control_simulation.py

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

    launch_control(baseconfig_filename='model.cfg',submit_runs=True,hyperparam_adjustments=hyperparam_adjustments)
