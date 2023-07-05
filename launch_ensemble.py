import os
import configparser

submit_runs=True

root_dir=os.path.expanduser('~/PlasmaEvolution')
baseconfig_filename='configs/default.cfg'

config=configparser.ConfigParser()
config.read(baseconfig_filename)
output_dir=config['model']['output_dir']
output_filename_base=config['model']['output_filename_base']
for ensemble_number in range(2):
    config_filename=os.path.join(output_dir,f'config{ensemble_number}')
    config['model']['output_filename_base']=output_filename_base+str(ensemble_number)
    with open(config_filename,'w') as f:
        config.write(f)
    slurm_text=f'''#!/bin/bash 

#SBATCH -N 1 
#SBATCH -c 8
#SBATCH --mem 48G
#SBATCH -G 1
#SBATCH -o {output_dir}log{ensemble_number}.out
#SBATCH -t 00:02:00

root_dir={root_dir}
conda activate torch
cd $root_dir
python train.py {config_filename}

exit'''
    slurm_filename=os.path.join(output_dir,f'job{ensemble_number}.slurm')
    with open(slurm_filename,'w') as f:
        f.write(slurm_text)
    if submit_runs:
        os.system(f'sbatch {slurm_filename}')
