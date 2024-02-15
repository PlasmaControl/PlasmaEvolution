import os
import configparser
import shutil

def launch_ensemble(baseconfig_filename='model.cfg',submit_runs=False,n_models=1,hyperparam_adjustments=[{}]):
    if n_models==1:
        ensemble_labels=['']
    else:
        ensemble_labels=[str(i) for i in range(n_models)]
    root_dir=os.path.dirname(os.path.realpath(__file__))
    config=configparser.ConfigParser()
    config.read(baseconfig_filename)
    for hyperparam_adjustment in hyperparam_adjustments:
        for category in hyperparam_adjustment:
            for hyperparam in hyperparam_adjustment[category]:
                config[category][hyperparam]=str(hyperparam_adjustment[category][hyperparam])
        output_dir=config['model']['output_dir']
        output_filename_base=config['model']['output_filename_base']
        # if ensembling, keep copy of config file without ensemble_number as the baseline used
        # for loading all the ensemble members
        if n_models>1:
            with open(os.path.join(output_dir,f'{output_filename_base}config'),'w') as f:
                config.write(f)
            #shutil.copyfile(baseconfig_filename, os.path.join(output_dir,f'{output_filename_base}config'))
        tune_model=config['model'].getboolean('tune_model',False)
        if tune_model:
            untuned_filename_base=config['model']['model_to_tune_filename_base']
        for ensemble_label in ensemble_labels:
            config_filename=os.path.join(output_dir,f'{output_filename_base}config{ensemble_label}')
            config['model']['output_filename_base']=f"{output_filename_base}{ensemble_label}"
            # comment this out or set to False if you want to launch ensemble off a single base model
            if tune_model:
                config['model']['model_to_tune_filename_base']=f"{untuned_filename_base}{ensemble_label}"
                print(f"Tuning from {config['model']['model_to_tune_filename_base']}")
            print(f"Training into {config['model']['output_filename_base']}")
            with open(config_filename,'w') as f:
                config.write(f)
            log_filename=os.path.join(output_dir,f'{output_filename_base}log{ensemble_label}.out')
            slurm_text=f'''#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem 48G
#SBATCH -G 1
#SBATCH -o {log_filename}
#SBATCH -t 2-00:00:00

root_dir={root_dir}
module load anaconda3/2022.5
conda activate torch
cd $root_dir
python -u ian_train.py {config_filename}

exit'''
            slurm_filename=os.path.join(output_dir,f'{output_filename_base}job{ensemble_label}.slurm')
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
    if False:
        hyperparam_adjustments=[]
        freeze_dic={'unfrozen': '', 'frozenEncoders': '\n'.join(['encoder','decoder']), 'frozenRNN': '\n'.join(['rnn'])}
        dataset_dic={'ip_0_900': 'ip_0_900'} 
        test_or_control='_tuned_on_data_only_'
        #dataset_dic={'ip_0_900': 'tetifromsim_astraInterpretiveAndTGLFNNip_0_1200_AND_ip_0_900'}
        #test_or_control=''
        frozen_profiles='\n'.join(['zipfit_trotfit_rho','zipfit_edensfit_rho','qpsi_EFIT01','zeff_rho'])
        rho_bdry_index=26
        #'ip_0_1200': 'tetifromsim_astraInterpretiveAndTGLFNNall_AND_ip_0_1200'}
        for which_dataset in dataset_dic:
            for which_freeze in freeze_dic:
                #resumed_dataset='_'.join(which_dataset.split('_')[-3:])
                hyperparam_adjustments.append(
                    {
                        'tuning': {'tune_model': True, 'resume_training': False, 'model_to_tune_filename_base': f'{which_dataset}NOdssdenest_RESUMED3',
                                   'frozen_layers': freeze_dic[which_freeze], 'masked_outputs': frozen_profiles, 'rho_bdry_index': rho_bdry_index},
                        'model': {'output_filename_base': f'surrogateHybrid{test_or_control}{which_dataset}{which_freeze}'},
                        'preprocess': {'preprocessed_data_filenamebase': f'/projects/EKOLEMEN/profile_predictor/final_paper/{dataset_dic[which_dataset]}'},
                        'optimization': {'n_epochs': 500, 'autoregression_num_steps': 10, 'nwarmup': 0,
                                         'autoregression_start_epoch': 0, 'autoregression_end_epoch': 0}
                    }
                )
    elif False:
        hyperparam_adjustments=[]
        preprocessed_filename_dic={#'astraInterpretiveAndTGLFNNip_0_900': 'astraInterpretiveAndTGLFNNip_0_900',
                                   #'astraInterpretiveAndTGLFNNip_0_1200': 'astraInterpretiveAndTGLFNNip_0_1200'}
                                   'astraInterpretiveAndTGLFNNall': 'astraInterpretiveAndTGLFNNall'}
        shared_calculations=[]
        calculations_dic={'noCalcs': [],
                          'withPredictive': ['PETOT_astrainterpretZIPFIT', 'PITOT_astrainterpretZIPFIT'],
                          'withInterpretive': ['TE_astrapredictTGLFNNZIPFIT', 'TI_astrapredictTGLFNNZIPFIT']}
        for which_trainset in preprocessed_filename_dic:
            for which_calcs in calculations_dic:
                hyperparam_adjustments.append(
                    {
                        'model': {'output_filename_base': f'{which_trainset}{which_calcs}'},
                        'preprocess': {'preprocessed_data_filenamebase': f'/projects/EKOLEMEN/profile_predictor/final_paper/{preprocessed_filename_dic[which_trainset]}'},
                        'inputs': {'calculations': '\n'.join(calculations_dic[which_calcs])}
                    }
                )
    elif True:
        hyperparam_adjustments=[]
        # dealing with training set
        #preprocessed_filename_dic={'augall': 'augall'} #{'augip_0_900': 'augip_0_900', 'augip_0_1200': 'augip_0_1200', 'augall': 'augall'}
        #preprocessed_filename_dic={'ip_0_1200': 'ip_0_1200', 'all': 'all'} #{'ip_0_900': 'ip_0_900', 'ip_0_1200': 'ip_0_1200', 'all': 'all'}
        #preprocessed_filename_dic={'aug900_d3d900': 'JOE_augip_0_900_AND_ip_0_900', 'augall_d3d900': 'JOE_augall_AND_ip_0_900'}
        preprocessed_filename_dic={'augall_d3d900': 'JOE_augall_AND_ip_0_900'}
        # dealing with inputs
        shared_actuators=['D_tot', 'pinj', 'tinj', 'ip', 'bt', 'ech_pwr_total', 'tribot_EFIT01', 'tritop_EFIT01', 'kappa_EFIT01', 'aminor_EFIT01', 'volume_EFIT01', 'rmaxis_EFIT01']
        shared_calculations=[]
        #actuators_dic={'WITHdssdenest': shared_actuators+['dssdenest'], 'NOdssdenest': shared_actuators} #{'NOdssdenest': shared_actuators}
        actuators_dic={'NOdssdenest': shared_actuators}
        #calculations_dic={'noSim': no_sim_calculations, 'withSim': sim_calculations}
        normalizations_dic={'NORMED': True, 'UNNORMED': False} #{'': False}
        #normalizations_dic={'withGBnormalization': True, 'noGBnormalization': False}
        hyperparam_adjustments=[]
        for which_trainset in preprocessed_filename_dic: #['allECH', 'ECH1MW', 'noECH']:
            for which_inputs in actuators_dic:
                for normalization_string in normalizations_dic:
                    hyperparam_adjustments.append(
                        {
                            ##### ADDING THIS TO START RERUNNING THEM #####
                            #'tuning': {'tune_model': True, 'resume_training': True, 'model_to_tune_filename_base': 'ip_0_900NOdssdenest_RESUMED3'}, #f'{which_trainset}{which_inputs}_RESUMED3'},
                            'model': {'output_filename_base': f'{which_trainset}{which_inputs}{normalization_string}'},
                            'preprocess': {'preprocessed_data_filenamebase': f'/projects/EKOLEMEN/profile_predictor/final_paper/{preprocessed_filename_dic[which_trainset]}',
                                           'use_fancy_normalization': normalizations_dic[normalization_string]},
                            'optimization': {'bucket_size': 1000},
                            'inputs':
                            {
                                'actuators': '\n'.join(actuators_dic[which_inputs])
                                #'calculations': '\n'.join(calculations_dic[which_inputs])
                            }
                        }
                    )
    launch_ensemble(baseconfig_filename='model.cfg',n_models=1,submit_runs=True,hyperparam_adjustments=hyperparam_adjustments)
