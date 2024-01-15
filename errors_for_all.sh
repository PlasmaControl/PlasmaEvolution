for dataset in ip_0_300 ip_0_600 ip_0_900 ip_0_1200;
do
    for inputs in noSim withSim;
    do
	python NEWmodelRollout.py /projects/EKOLEMEN/profile_predictor/paper_models/"$dataset""$inputs"config
    done
done
