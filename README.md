This repo trains and analyzes neural nets for predicting how a tokamak (fusion reactor) plasma will evolve in time given an initial condition and user-specified actuator trajectories. It is a (pytorch-based) cleanup of the (tensorflow-based) [plasma-profile-predictor](https://github.com/PlasmaControl/plasma-profile-predictor), which was described in our [2021 Nuclear Fusion paper](https://doi.org/10.1088/1741-4326/abe08d).

Generate an h5 file with [data-fetching repo](https://github.com/PlasmaControl/data-fetching)

In configs/default.cfg point raw_data_filename to the generated h5 file. Then change preprocessed_data_filename_base to a "base" name for writing processed data. Run preprocess_data.py, which will generate the basename with _train.pkl, _val.pkl, and _test.pkl appended. Change output_dir in the config file to where you want to dump a model, then run python ian_train.py to train a model to go there. To train a full ensemble of models (submitting them to slurm on traverse) do python launch_ensemble.py which will train 10 with 0,...,9 appended to the end.

Run modelRollout.py {config_filename} (where config_filename is the full path to the config file corresponding to the model) to plot the outputs. Set plot_ensemble to True or False depending on whether you're doing ensemble modeling or one model at a time.

You can also use python modelStats.py {config_filename} . Again, set plot_ensemble to True or False

For pytorch environment setup on PPPL/Princeton's Traverse cluster along with a ton of other helpful info and examples, see [researchcomputing.princeton.edu](https://researchcomputing.princeton.edu/pytorch). h5py is also required for reading the h5 dataset. As of February 2023, I personally use
    module load anaconda3/2022.5
    conda create --name torch --channel "https://opence.mit.edu/#/" "pytorch==1.12*=cuda11*" torchvision
    conda install -c anaconda h5py
    conda activate torch


And of course reload anaconda and activate this environment every time you go to run the code.


-------- TO HELP TEST ---------
Set train_shots, val_shots, and test_shots to a small number of shots each