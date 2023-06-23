import numpy as np
import copy
import os

def get_ensemble_info(modelDir):
    ensemble_weight_matrices=[]
    ensemble_bias_arrays=[]
    ensemble_steepnesses=[]
    ensemble_activations=[]
    for filename in os.listdir(modelDir):
        with open(os.path.join(modelDir, filename),'r') as f:
            params={}
            for line in f.readlines()[1:]:
                (variable,data)=line.split('=')
                params[variable]=data
        num_layers=int(params['num_layers'])
        def get_arr_from_str(string, dtype=float):
            ret=[dtype(elem) for elem in string.split()]
            if dtype is str:
                ret=[key.replace("'","") for key in ret]
            return ret
        def get_2darr_from_str(string):
            arr=[]
            for line in string.split('(')[1:]:
                line=line.split(')')[0]
                arr.append([float(elem) for elem in line.split(',')])
            return arr
        layer_sizes=get_arr_from_str(params['layer_sizes'],dtype=int)
        input_names=get_arr_from_str(params['input_names'],dtype=str)
        scale_mean_in=get_arr_from_str(params['scale_mean_in'])
        scale_mean_out=get_arr_from_str(params['scale_mean_out'])
        scale_deviation_in=get_arr_from_str(params['scale_deviation_in'])
        scale_deviation_out=get_arr_from_str(params['scale_deviation_out'])
        neurons=get_2darr_from_str(params['neurons (num_inputs, activation_function, activation_steepness)'])
        connections=get_2darr_from_str(params['connections (connected_to_neuron, weight)'])
        neur=np.array(neurons)
        conn=np.array(connections)
        weight_matrices=[]
        bias_arrays=[]
        steepnesses=[]
        activations=[]
        connection_ind=0
        neuron_ind=layer_sizes[0]
        for i in range(len(layer_sizes)-1):
            layer_size=layer_sizes[i]
            # libfann documentation: "There will be a bias neuron in each layer (except the output layer),
            # and this bias neuron will be connected to all neurons in the next layer.
            # When running the network, the bias nodes always emits 1."
            next_layer_size=layer_sizes[i+1]-1
            next_connection_ind=connection_ind+next_layer_size*layer_size
            relevant_connections=conn[connection_ind:next_connection_ind,1].reshape((next_layer_size,layer_size))
            activations.append(neur[neuron_ind,1])
            steepnesses.append(neur[neuron_ind,2])
            neuron_ind+=layer_sizes[i+1]
            connection_ind=next_connection_ind
            weight_matrices.append(relevant_connections[:,:-1])
            bias_arrays.append(relevant_connections[:,-1])
        ensemble_weight_matrices.append(weight_matrices)
        ensemble_bias_arrays.append(bias_arrays)
        ensemble_activations.append(activations)
        ensemble_steepnesses.append(steepnesses)
    ensemble_info={'input_names': input_names,'scale_mean_in': scale_mean_in, 'scale_mean_out': scale_mean_out,
                   'scale_deviation_in': scale_deviation_in, 'scale_deviation_out': scale_deviation_out,
                   'ensemble_weight_matrices': ensemble_weight_matrices, 'ensemble_bias_arrays': ensemble_bias_arrays,
                   'ensemble_activations': ensemble_activations, 'ensemble_steepnesses': ensemble_steepnesses}
    return ensemble_info

def evaluate_model(test_input, ensemble_info):
    norm_input=(test_input-ensemble_info['scale_mean_in'])/ensemble_info['scale_deviation_in']
    all_norm_outputs=[]
    #https://github.com/libfann/fann/blob/master/src/include/fann_data.h#L204
    def linear_activation(arr, steepness):
        return steepness*arr
    def tanh_activation(arr,steepness):
        return np.tanh(steepness*arr)
    activation_function_mapping={0: linear_activation, 5: tanh_activation}
    for which_model in range(len(ensemble_info['ensemble_weight_matrices'])):
        h=copy.deepcopy(norm_input)
        for which_layer in range(len(ensemble_info['ensemble_weight_matrices'][which_model])):
            activation_function=activation_function_mapping[ensemble_info['ensemble_activations'][which_model][which_layer]]
            h=np.matmul(ensemble_info['ensemble_weight_matrices'][which_model][which_layer],h)+ensemble_info['ensemble_bias_arrays'][which_model][which_layer]
            h=activation_function(h,ensemble_info['ensemble_steepnesses'][which_model][which_layer])
        all_norm_outputs.append(h)
    # "'OUT_p_E1_0' 'OUT_p_E1_2' 'OUT_wid_E1_0' 'OUT_wid_E1_2'\n"
    all_norm_outputs=np.array(all_norm_outputs)*ensemble_info['scale_deviation_out']+ensemble_info['scale_mean_out']
    return np.mean(all_norm_outputs,axis=0), np.std(all_norm_outputs,axis=0)
