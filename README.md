This repo trains and analyzes neural nets for predicting how a tokamak (fusion reactor) plasma will evolve in time given an initial condition and user-specified actuator trajectories. It is a (pytorch-based) cleanup of the (tensorflow-based) [plasma-profile-predictor](https://github.com/PlasmaControl/plasma-profile-predictor), which was described in our [2021 Nuclear Fusion paper](https://doi.org/10.1088/1741-4326/abe08d).

Generate an h5 file with [data-fetching repo](https://github.com/PlasmaControl/data-fetching)

-------- TO TRAIN A MODEL ---------
In configs/default.cfg point raw_data_filename to the generated h5 file. Then change preprocessed_data_filename_base to a "base" name for writing processed data. Run preprocess_data.py, which will generate the basename with _train.pkl, _val.pkl, and _test.pkl appended. Change output_dir in the config file to where you want to dump a model, then run python ian_train.py to train a model to go there. To train a full ensemble of models (submitting them to slurm on traverse) do python launch_ensemble.py which will train 10 with 0,...,9 appended to the end. Use modelStats.py {config_filename} to plot training losses.

-------- TO CREATE AND VISUALIZE MODEL OUTPUTS ---------
Run SimpleModelRollout.py {config_filename} (where config_filename is the full path to the config file corresponding to the model) to create a pickle file with the predicted profiles. Set plot_ensemble to True or False depending on whether you're doing ensemble modeling or one model at a time. To visualize the predictions, use prediction_plotter.ipynb

-------- TO HELP TEST ---------
Set train_shots, val_shots, and test_shots to a small number of shots each

-------- TO TUNE MODELS -------
Training takes a long time with curriculum learning.
If you have a good model with the right inputs and want to tune it on simulation or other-machine data, or overfit it to data (e.g. for developing control for a specific scenario), copy the config file from the model you like, then
1) set tune_model to true
2) set the model_to_tune_filename_base to the output_filename_base (i.e. you're tuning the good model)
3) set the output_filename_base to your new model name
If you only want to tune certain layers add the layer names as a list. In torch layer names correspond to named class variables (self.rnn --> rnn, self.encoder --> encoder, self.decoder --> decoder) and can be found with
for name, child in model.named_children():
    print(name)
4) for most cases set autoregression_start_epoch and autoregression_end_epoch to 0 since you don't want to go back to the beginning of the curriculum learning, you want to start from what you already have
5) for some cases (e.g. for training on simulations where you only confidently can predict out to rho=0.8 and you only predict a subset of the profiles like TE and TI) add a list of the profiles you don't want to consider to masked_outputs and set tuning-->rho_bdry_index to the index of the boundary (e.g. rho=0.8 for nx=33 would be 0.8*33 ~ 26). This will mask those outputs during training.

-------- LRAN TRAINING AND CONTROL ---------
To train a model with latent linear dynamics (LRAN) to be used to control applications, change the model.cfg model_type from IanRNN to HiroLRAN and repeat training steps above. To launch a control simulation, where the controller makes MPC actuator decisions using HiroLRAN and the profile evolution is simulated using IanRNN, use control_simulation.py. To visualize these results use control_plotter.ipynb. 

-------- ENVIRONMENT SETUP ---------
For pytorch environment setup on PPPL/Princeton's Traverse cluster along with a ton of other helpful info and examples, see [researchcomputing.princeton.edu](https://researchcomputing.princeton.edu/pytorch). h5py is also required for reading the h5 dataset. As of February 2023, I personally use
    module load anaconda3/2022.5
    conda create --name torch --channel "https://opence.mit.edu/#/" "pytorch==1.12*=cuda11*" torchvision
    conda install -c anaconda h5py
    conda activate torch

And of course reload anaconda and activate this environment every time you go to run the code.