import os
import configparser
import shutil
submit_runs=True

root_dir=os.path.expanduser('~/PlasmaEvolution')
baseconfig_filename='preprocess.cfg'
config=configparser.ConfigParser()
config.read(baseconfig_filename)
output_dir=config['logistics']['output_dir']
output_filename_base=config['logistics']['output_filename_base']

config_filename=os.path.join(output_dir,f'{output_filename_base}config')
shutil.copyfile(baseconfig_filename, config_filename)
# test_index=config_filename['shot'].get('test_index',0)
# val_index_choices=[num for num in range(10) if num!=test_index]
# config_filename['shots']['val_index']=np.random.choice(val_index_choices)
#with open(config_filename,'w') as f:
#    config.write(f)
log_filename=os.path.join(output_dir,f'{output_filename_base}log.out')
slurm_text=f'''#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem 48G
#SBATCH -o {log_filename}
#SBATCH -t 12:00:00

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
