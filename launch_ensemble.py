import os
import configparser
import shutil
submit_runs=True

root_dir=os.path.expanduser('~/PlasmaEvolution')
baseconfig_filename='configs/default.cfg'
config=configparser.ConfigParser()
config.read(baseconfig_filename)
output_dir=config['model']['output_dir']
output_filename_base=config['model']['output_filename_base']
tune_model=config['model'].getboolean('tune_model',False)
if tune_model:
    untuned_filename_base=config['model']['model_to_tune_filename_base']

#keep copy of config file without ensemble_number for testing
shutil.copyfile(baseconfig_filename, os.path.join(output_dir,f'{output_filename_base}config'))
for ensemble_number in range(10):
    config_filename=os.path.join(output_dir,f'{output_filename_base}config{ensemble_number}')
    config['model']['output_filename_base']=f"{output_filename_base}{str(ensemble_number)}"
    # comment this out or set to False if you want to launch ensemble off a single base model
    if tune_model:
        config['model']['model_to_tune_filename_base']=f"{untuned_filename_base}{str(ensemble_number)}"
        print(f"Tuning from {config['model']['model_to_tune_filename_base']}")
    print(f"Training into {config['model']['output_filename_base']}")
    with open(config_filename,'w') as f:
        config.write(f)
    log_filename=os.path.join(output_dir,f'{output_filename_base}log{ensemble_number}.out')
    slurm_text=f'''#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem 48G
#SBATCH -G 1
#SBATCH -o {log_filename}
#SBATCH -t 05:00:00

root_dir={root_dir}
module load anaconda
conda activate torch
cd $root_dir
python -u ian_train.py {config_filename}

exit'''
    slurm_filename=os.path.join(output_dir,f'{output_filename_base}job{ensemble_number}.slurm')
    with open(slurm_filename,'w') as f:
        f.write(slurm_text)
    if submit_runs:
        os.system(f'sbatch {slurm_filename}')
