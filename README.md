This repo trains and analyzes neural nets for predicting how a tokamak (fusion reactor) plasma will evolve in time given an initial condition and user-specified actuator trajectories. It is a (pytorch-based) cleanup of the (tensorflow-based) [plasma-profile-predictor](https://github.com/PlasmaControl/plasma-profile-predictor), which was described in our [2021 Nuclear Fusion paper](https://doi.org/10.1088/1741-4326/abe08d).

In train.py point the data_filename to an h5 file generated from [data-fetching repo](https://github.com/PlasmaControl/data-fetching), then run test case with
    python train.py

For pytorch environment setup on PPPL/Princeton's Traverse cluster along with a ton of other helpful info and examples, see [researchcomputing.princeton.edu](https://researchcomputing.princeton.edu/pytorch). h5py is also required for reading the h5 dataset. As of February 2023, I personally use
    module load anaconda3/2022.5
    conda create --name torch --channel "https://opence.mit.edu/#/" "pytorch==1.12*=cuda11*" torchvision
    conda install -c anaconda h5py
    conda activate torch

And of course reload anaconda and activate this environment every time you go to run the code.